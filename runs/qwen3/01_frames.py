"""Frames for the VLM ladder on the OFFICIAL anticipation set, camera C10119 (same segments as runs/tsm/01_bank.py).
Per segment (cut = start_frame - 30, i.e. 1 s before the action):
  recent : 4 frames at -1.5, -1.0, -0.5, 0 s before the cut
  hist   : 8 frames over the last 60 s (-60, -52.5, ..., -7.5 s), dropped if before the video start
Frames are shared by overlapping segments, so each is saved once: <out>/frames/<recording>/<frame30:06d>.jpg,
448 x 252 (16:9), JPEG quality 90. Index: <out>/<split>.json = [{id, seq, start, y:[action, verb, object],
recent:[...], hist:[...]}] with 30-fps frame numbers. Videos are streamed one recording at a time and deleted.
usage: python 01_frames.py /workspace/data/a101_anticip/data/CSVs /workspace/data/a101_vlm [--limit 2]
"""
import argparse, csv, json, os, queue, shutil, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HF_HOME", "/workspace/hf")
import decord
from PIL import Image
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("csvs"); ap.add_argument("out")
ap.add_argument("--view", default="C10119_rgb")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
REPO = "cvml-nus/assembly101"
TMP = os.path.join(a.out, "_tmp")
FR = os.path.join(a.out, "frames")
os.makedirs(FR, exist_ok=True)
REC = [-45, -30, -15, 0]                                  # 30-fps offsets from the cut: -1.5, -1, -0.5, 0 s
HIST = [-1800 + 225 * i for i in range(8)]                # -60 s ... -7.5 s


def rows_of(split):
    out = []
    if split == "train":
        with open(os.path.join(a.csvs, "train.csv")) as f:
            for i, r in enumerate(csv.DictReader(f)):
                if r["video"].endswith(f"/{a.view}.mp4"):
                    out.append({"id": f"tr{i}", "seq": r["video"].split("/")[0], "start": int(r["start_frame"]),
                                "y": [int(r["action_id"]), int(r["verb_id"]), int(r["noun_id"])]})
    else:
        with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
            for r in csv.DictReader(f):
                out.append({"id": r["id"], "seq": r["video"].split("/")[0], "start": int(r["start"]),
                            "y": [int(r["action"]), int(r["verb"]), int(r["noun"])]})
    for q in out:
        cut = q["start"] - 30
        q["recent"] = [max(0, cut + o) for o in REC]
        q["hist"] = [cut + o for o in HIST if cut + o >= 0]
    return out


segs = {s: rows_of(s) for s in ("train", "validation")}
for s, rows in segs.items():
    json.dump(rows, open(os.path.join(a.out, f"{s}.json"), "w"))
need = {}
for rows in segs.values():
    for q in rows:
        need.setdefault(q["seq"], set()).update(q["recent"] + q["hist"])
seqs = sorted(need)
todo = [s for s in seqs if not os.path.exists(os.path.join(FR, s, ".done"))]
if a.limit:
    todo = todo[: a.limit]
print({s: len(r) for s, r in segs.items()}, f"segments | {len(seqs)} recordings, {sum(len(v) for v in need.values())} frames"
      f" | {len(todo)} recordings to do", flush=True)

ready = queue.Queue(maxsize=2)


def downloader():
    for s in todo:
        p = None
        for attempt in range(3):
            try:
                p = hf_hub_download(REPO, f"recordings/{s}/{a.view}.mp4", repo_type="dataset", local_dir=os.path.join(TMP, s))
                break
            except Exception as e:
                print(f"download {s} failed ({e}), retry {attempt + 1}", flush=True); time.sleep(10)
        ready.put((s, p))
    ready.put(None)


threading.Thread(target=downloader, daemon=True).start()
t0, done = time.time(), 0
while True:
    item = ready.get()
    if item is None:
        break
    s, path = item
    if path is None:
        continue
    vr = decord.VideoReader(path, width=448, height=252, num_threads=8)
    fps, n = vr.get_avg_fps(), len(vr)
    d = os.path.join(FR, s); os.makedirs(d, exist_ok=True)
    fr30 = sorted(need[s])
    idx = [min(n - 1, round(f * fps / 30)) for f in fr30]
    for c in range(0, len(idx), 64):
        arr = vr.get_batch(idx[c:c + 64]).asnumpy()
        for f, im in zip(fr30[c:c + 64], arr):
            Image.fromarray(im).save(os.path.join(d, f"{f:06d}.jpg"), quality=90)
    open(os.path.join(d, ".done"), "w").close()
    del vr
    shutil.rmtree(os.path.join(TMP, s), ignore_errors=True)
    done += 1
    el = time.time() - t0
    print(f"[{done}/{len(todo)}] {s}: {len(fr30)} frames | {el / 60:.1f} min, ~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
print("done")
