"""Hand-pose tokens for anticipation (camera C10119 segments; poses are Assembly101's released 3D hand poses, 60 fps,
reconstructed from the multi-view rig, so rows using them are "+ hand pose", not single-camera).
1. download poses@60fps/<recording>.json from the HF dataset, convert to <out>/npz/<recording>.npz
   (joints [T, 2, 21, 3] float16 in mm, NaN where a hand is missing) and delete the JSON (resumable).
2. tokens per segment (cut = start - 30 at 30 fps = 60 fps index 2 * cut):
   past   16 steps over the last 4 s (every 0.25 s)   -> student input
   future  8 steps over the next 2 s (every 0.25 s)    -> teacher only
   each step: per hand [wrist position (relative to the recording's mean wrist, /100) | 20 joints relative to the
   wrist (/100) | wrist velocity over 0.25 s | present flag] -> 2 x (3 + 60 + 3 + 1) = 134 values
Saves <out>/hand_bank.pt = {split: {"past": [n, 16, 134] fp16, "fut": [n, 8, 134] fp16}}, segment order as the TSM bank.
usage: python 15_hand_bank.py /workspace/data/a101_anticip/data/CSVs /workspace/data/a101_poses
"""
import argparse, csv, json, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np
import torch
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("csvs"); ap.add_argument("out")
ap.add_argument("--view", default="C10119_rgb")
a = ap.parse_args()
NPZ = os.path.join(a.out, "npz"); os.makedirs(NPZ, exist_ok=True)


def rows_of(split):
    out = []
    if split == "train":
        with open(os.path.join(a.csvs, "train.csv")) as f:
            for r in csv.DictReader(f):
                if r["video"].endswith(f"/{a.view}.mp4"):
                    out.append((r["video"].split("/")[0], int(r["start_frame"]), int(r["end_frame"])))
    else:
        with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
            for r in csv.DictReader(f):
                out.append((r["video"].split("/")[0], int(r["start"]), int(r["end"])))
    return out


segs = {s: rows_of(s) for s in ("train", "validation")}
seqs = sorted({q for rows in segs.values() for q, _, _ in rows})
last_end = {}
for rows in segs.values():
    for q, _, e in rows:
        last_end[q] = max(last_end.get(q, 0), e)
print({s: len(r) for s, r in segs.items()}, f"segments in {len(seqs)} recordings", flush=True)

# ------------------------------------------------------------------ 1. download + convert
t0, missing = time.time(), []
for k, seq in enumerate(seqs):
    dst = os.path.join(NPZ, f"{seq}.npz")
    if os.path.exists(dst):
        continue
    try:
        p = hf_hub_download("cvml-nus/assembly101", f"poses@60fps/{seq}.json", repo_type="dataset", local_dir=os.path.join(a.out, "_tmp"))
    except Exception as e:
        missing.append(seq); print(f"no poses for {seq} ({type(e).__name__})", flush=True); continue
    d = json.load(open(p))
    T = max(x["frame_index"] for x in d) + 1
    J = np.full((T, 2, 21, 3), np.nan, np.float32)
    for x in d:
        for h, pts in x.get("landmarks", {}).items():
            if int(h) < 2 and pts is not None and len(pts) == 21:
                J[x["frame_index"], int(h)] = np.asarray(pts, np.float32)
    np.savez_compressed(dst, joints=J.astype(np.float16))
    os.remove(p)
    if k < 5:     # alignment check: pose frames at 60 fps vs the last annotated 30-fps frame
        print(f"[alignment] {seq}: {T} pose frames = {T / 60:.0f} s | last annotated end frame {last_end[seq]} = "
              f"{last_end[seq] / 30:.0f} s | first timestamp {d[0].get('timestamp')}", flush=True)
    if k % 25 == 0:
        print(f"poses {k + 1}/{len(seqs)} | {(time.time() - t0) / 60:.1f} min | {seq}: {T} frames, "
              f"hands present {np.isfinite(J[:, :, 0, 0]).mean(0).round(2).tolist()}", flush=True)
print(f"poses done: {len(seqs) - len(missing)} recordings, {len(missing)} missing | {(time.time() - t0) / 60:.1f} min", flush=True)

# ------------------------------------------------------------------ 2. tokens
PAST = [-240 + 15 * i for i in range(16)]       # 60 fps offsets from the cut: -4.0 ... -0.25 s
FUT = [15 * (i + 1) for i in range(8)]          # +0.25 ... +2.0 s


def step_feats(J, idx, center):
    """J [T, 2, 21, 3]; idx list of 60-fps frames -> [len(idx), 134]."""
    T = len(J)
    out = np.zeros((len(idx), 2, 67), np.float32)
    for n, t in enumerate(idx):
        t = min(max(t, 0), T - 1); tp = min(max(t - 15, 0), T - 1)
        for h in range(2):
            j, jp = J[t, h], J[tp, h]
            if not np.isfinite(j).all():
                continue
            w = j[0]
            vel = (w - jp[0]) / 100 if np.isfinite(jp).all() else np.zeros(3)
            out[n, h] = np.concatenate([(w - center) / 100, ((j[1:] - w) / 100).ravel(), vel, [1.0]])
    return out.reshape(len(idx), 134)


bank = {}
for s, rows in segs.items():
    P = np.zeros((len(rows), 16, 134), np.float16); Fu = np.zeros((len(rows), 8, 134), np.float16)
    cur, J, center = None, None, None
    for i, (seq, start, _) in enumerate(rows):
        if seq != cur:
            cur = seq
            f = os.path.join(NPZ, f"{seq}.npz")
            J = np.load(f)["joints"].astype(np.float32) if os.path.exists(f) else None
            if J is not None:
                wr = J[:, :, 0].reshape(-1, 3); center = np.nanmean(wr[np.isfinite(wr).all(1)], 0)
        if J is None:
            continue
        c60 = 2 * (start - 30)
        P[i] = step_feats(J, [c60 + o for o in PAST], center)
        Fu[i] = step_feats(J, [c60 + o for o in FUT], center)
    bank[s] = {"past": torch.from_numpy(P), "fut": torch.from_numpy(Fu)}
    pres = P[:, :, 66::67].astype(np.float32).mean()
    print(f"{s}: hand tokens built | share of steps with a hand present {pres:.2f}", flush=True)
torch.save(bank, os.path.join(a.out, "hand_bank.pt"))
print(f"saved {os.path.join(a.out, 'hand_bank.pt')} | {(time.time() - t0) / 60:.1f} min")
