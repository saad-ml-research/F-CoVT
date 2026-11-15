"""TSM feature bank on the OFFICIAL Assembly101 anticipation set, camera C10119 (released TSM features, 2048-D per
frame at 30 fps, fine-tuned on Assembly101 action recognition). Protocol as the official TempAgg code:
  cut = start_frame - 30 (1 s before the action), features used up to the cut, max-pooled over short windows,
  lmdb key = '<recording>/C10119_rgb/C10119_rgb_<frame + 1 :010d>.jpg'.
Per segment we store (all max-pooled windows, fp16):
  recent : the last 2 s before the cut, 8 windows of 0.25 s                      (student input)
  hist   : the last 60 s before the cut, 30 windows of 2 s (+ mask: before the video start = 0)  (history)
  fut    : the next 2 s after the cut, 8 windows of 0.25 s (the action starts at window 4)  (training only)
Labels: official 17 verbs / 90 objects / 1,063 actions. Train = train.csv rows of this camera; validation =
validation_challenge.csv (13,494 segments; segments with a cut before 2 s are KEPT, unlike TempAgg which drops them
and scores them as misses).
The lmdb is read once, recording by recording (keys are sorted), so every frame is read once.
usage: python 01_bank.py /workspace/data/a101_feats/TSM_features/C10119_rgb /workspace/data/a101_anticip/data/CSVs
                         /workspace/outputs/tsm_bank [--limit 2]
"""
import argparse, collections, csv, os, time
import numpy as np
import lmdb
import torch

ap = argparse.ArgumentParser()
ap.add_argument("lmdb"); ap.add_argument("csvs"); ap.add_argument("out")
ap.add_argument("--view", default="C10119_rgb")
ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N recordings")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FPS = 30
REC = np.linspace(-60, 0, 9).round().astype(int)                     # 8 windows over the last 2 s
HIST = np.linspace(-1800, 0, 31).round().astype(int)                  # 30 windows of 2 s over the last 60 s
FUT = np.linspace(1, 60, 9).round().astype(int)                       # 8 windows over the next 2 s


def rows_of(split):
    out = []
    if split == "train":
        with open(os.path.join(a.csvs, "train.csv")) as f:
            for r in csv.DictReader(f):
                if r["video"].endswith(f"/{a.view}.mp4"):
                    out.append((r["video"].split("/")[0], int(r["start_frame"]), int(r["end_frame"]),
                                int(r["action_id"]), int(r["verb_id"]), int(r["noun_id"])))
    else:
        with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
            for r in csv.DictReader(f):
                out.append((r["video"].split("/")[0], int(r["start"]), int(r["end"]),
                            int(r["action"]), int(r["verb"]), int(r["noun"])))
    return out


segs = {s: rows_of(s) for s in ("train", "validation")}
by_seq = collections.defaultdict(list)
for s, rows in segs.items():
    for i, r in enumerate(rows):
        by_seq[r[0]].append((s, i))
print({s: len(r) for s, r in segs.items()}, f"segments in {len(by_seq)} recordings", flush=True)

store = {s: {"recent": np.zeros((len(r), 8, 2048), np.float16), "hist": np.zeros((len(r), 30, 2048), np.float16),
             "hist_mask": np.zeros((len(r), 30), bool), "fut": np.zeros((len(r), 8, 2048), np.float16),
             "found": np.zeros(len(r), bool)} for s, r in segs.items()}


def pool(F, edges):
    """max over frames edges[j]..edges[j+1] (inclusive, clamped to the video); returns [len-1, 2048], valid mask."""
    n = len(F)
    out = np.zeros((len(edges) - 1, F.shape[1]), np.float32); ok = np.zeros(len(edges) - 1, bool)
    for j in range(len(edges) - 1):
        lo, hi = max(0, edges[j]), min(n - 1, edges[j + 1])
        if hi >= lo:
            out[j] = F[lo:hi + 1].max(0); ok[j] = True
    return out, ok


def process(seq, frames):
    n = max(frames) + 1
    F = np.zeros((n, 2048), np.float32)
    for k, v in frames.items():
        F[k] = v
    for s, i in by_seq.get(seq, []):
        cut = segs[s][i][1] - FPS
        st = store[s]
        st["recent"][i] = pool(F, cut + REC)[0]
        h, ok = pool(F, cut + HIST); st["hist"][i] = h; st["hist_mask"][i] = ok
        st["fut"][i] = pool(F, cut + FUT)[0]
        st["found"][i] = True


env = lmdb.open(a.lmdb, readonly=True, lock=False, readahead=True)
t0, done, cur, frames = time.time(), 0, None, {}
with env.begin() as txn:
    for key, val in txn.cursor():
        k = key.decode()
        seq = k.split("/")[0]
        if seq != cur:
            if cur is not None and cur in by_seq:
                process(cur, frames); done += 1
                if done % 10 == 0:
                    print(f"{done}/{len(by_seq)} recordings | {(time.time() - t0) / 60:.1f} min", flush=True)
                if a.limit and done >= a.limit:
                    cur = None; break
            cur, frames = seq, {}
        if seq in by_seq:
            frames[int(k.rsplit("_", 1)[1].split(".")[0]) - 1] = np.frombuffer(val, np.float32)
    if cur is not None and cur in by_seq:
        process(cur, frames); done += 1

for s, rows in segs.items():
    st = store[s]
    keep = st["found"] if a.limit else np.ones(len(rows), bool)
    missing = int((~st["found"]).sum())
    y = np.array([[r[3], r[4], r[5]] for r in rows])
    torch.save({"recent": torch.from_numpy(st["recent"][keep]), "hist": torch.from_numpy(st["hist"][keep]),
                "hist_mask": torch.from_numpy(st["hist_mask"][keep]), "fut": torch.from_numpy(st["fut"][keep]),
                "y": torch.from_numpy(y[keep]), "seq": [r[0] for r, k in zip(rows, keep) if k],
                "start": torch.tensor([r[1] for r, k in zip(rows, keep) if k]),
                "windows": {"recent_s": 0.25, "hist_s": 2.0, "fut_s": 0.25}},
               os.path.join(a.out, f"{s}.pt"))
    print(f"saved {s}: {int(keep.sum())} segments ({missing} without features"
          f"{' - zeros kept' if not a.limit and missing else ''})", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min")
