"""Build the training subset from Future-L1-50K parquet shards WITHOUT keeping videos.
For each shard: download (2 GB) -> for each row, decode the 8-frame pool exactly like
Future-L1's loader -> save f1.jpg..f8.jpg -> delete the shard. Writes a TwiFF-format JSON
whose "video" field points to the frame folder (read by our fl1_frames_dir patch).
usage: python 07_fl1_frames.py --shards 26 --out /workspace/data/fl1_10k
26 shards x ~387 rows = ~10K samples. Resumable (skips finished shards)."""
import argparse, io, json, os, tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import decord
from PIL import Image
from huggingface_hub import hf_hub_download

def _get_frame_indices_uni(num_frames, vlen, start_frames=1, end_frames=1):
    # copied verbatim from Future-L1 src/dataset/twiff_sft_dataset.py
    if start_frames + 1 >= vlen:
        start_frames = 0
    if vlen - end_frames - 1 <= start_frames:
        end_idx = vlen - 1
    else:
        end_idx = vlen - end_frames - 1
    acc_samples = min(num_frames - 2, end_idx - start_frames)
    intervals = np.linspace(start=start_frames + 1, stop=end_idx, num=acc_samples + 1).astype(int)
    middle = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(len(intervals) - 1)]
    frame_indices = [start_frames] + middle + [end_idx]
    if len(frame_indices) < num_frames:
        padded = [frame_indices[-1]] * num_frames
        padded[: len(frame_indices)] = frame_indices
        frame_indices = padded
    return frame_indices

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=26)
ap.add_argument("--out", default="/workspace/data/fl1_10k")
ap.add_argument("--max_side", type=int, default=640)
a = ap.parse_args()
out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
tmp = Path(f"/workspace/data/_tmp_shards/{os.getpid()}"); tmp.mkdir(parents=True, exist_ok=True)
lock = out / "RUNNING.pid"
if lock.exists() and Path(f"/proc/{lock.read_text().strip()}").exists():
    raise SystemExit(f"another 07_fl1_frames.py is running (pid {lock.read_text().strip()}); stop it first")
lock.write_text(str(os.getpid()))
done_file = out / "done_shards.txt"
done = set(done_file.read_text().split()) if done_file.exists() else set()
samples_file = out / "samples.jsonl"

for k in range(a.shards):
    name = f"data/train-{k:05d}-of-00129.parquet"
    if name in done:
        continue
    p = hf_hub_download("Eurayka/Future-L1-50K", name, repo_type="dataset", local_dir=tmp)
    pf = pq.ParquetFile(p)
    n_ok = n_bad = 0
    with open(samples_file, "a") as fo:
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                fdir = out / "frames" / row["id"]
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp") as tf:
                        tf.write(row["video"]["bytes"]); tf.flush()
                        vr = decord.VideoReader(tf.name, num_threads=1)
                        pool = _get_frame_indices_uni(8, len(vr), 1, 1)
                        frames = vr.get_batch(pool).asnumpy()
                    fdir.mkdir(parents=True, exist_ok=True)
                    for i, f in enumerate(frames, 1):
                        im = Image.fromarray(f)
                        im.thumbnail((a.max_side, a.max_side))
                        im.save(fdir / f"f{i}.jpg", quality=90)
                    rec = {k2: v for k2, v in row.items() if k2 != "video"}
                    rec["video"] = str(fdir)
                    rec["orig_video"] = row["video"]["path"]
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_ok += 1
                except Exception as e:
                    n_bad += 1
    Path(p).unlink(missing_ok=True)
    with open(done_file, "a") as fd:
        fd.write(name + "\n")
    print(f"{name}: {n_ok} ok, {n_bad} failed", flush=True)

recs = list({r["id"]: r for r in (json.loads(l) for l in open(samples_file))}.values())  # dedupe by id
lock.unlink(missing_ok=True)
ov = Path("/workspace/data/overlap_train_ids.txt")   # from 09_overlap_check.py
if ov.exists():
    bad = set(ov.read_text().split())
    n0 = len(recs); recs = [r for r in recs if r["id"] not in bad]
    print(f"dropped {n0 - len(recs)} training samples overlapping FutureBench videos")
json.dump(recs, open(out / "train.json", "w"), ensure_ascii=False)
print(f"wrote {len(recs)} samples -> {out/'train.json'}")
