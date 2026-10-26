"""Precompute SAM masks (the slow expert) for F-CoVT targets, exactly like CoVT's
get_sam_mask_improved: image resized to 256x256, SamAutomaticMaskGenerator (SAM v1 ViT-H, default
settings), keep top-8 masks by predicted_iou * stability_score.
Order: all FUTURE frames (reasoning_image) first, then CURRENT frames (image) - resumable, skips done files.
Output: /workspace/data/fl1_10k/sam_masks/<id>/f<k>.npz  {masks: packed bool (n,256,256), n, score}
DINO and DepthAnything targets are fast forward passes and are computed online during training."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.path.insert(0, "/workspace/CoVT/train/src")           # CoVT's SAM fork (same code CoVT trains with)
from anchors.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/data/fl1_10k/train.json")
ap.add_argument("--out", default="/workspace/data/fl1_10k/sam_masks")
ap.add_argument("--ckpt", default="/workspace/CoVT/train/src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth")
ap.add_argument("--current", type=int, default=1, help="also do the current (question) frames, after all future ones")
ap.add_argument("--points_per_batch", type=int, default=256, help="speed only; does not change results")
ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (for a speed test)")
ap.add_argument("--shard", default="0/1", help="i/n: this worker takes every n-th frame starting at i")
a = ap.parse_args()

data = json.load(open(a.data))
jobs = [(d["id"], d["video"], k) for d in data for k in d["reasoning_image"]]
if a.current:
    jobs += [(d["id"], d["video"], k) for d in data for k in d["image"]]
seen = set(); jobs = [j for j in jobs if not ((j[0], j[2]) in seen or seen.add((j[0], j[2])))]
if a.limit:
    jobs = jobs[: a.limit]
si, sn = map(int, a.shard.split("/")); jobs = jobs[si::sn]
out = Path(a.out)
todo = [j for j in jobs if not (out / j[0] / f"f{j[2]}.npz").exists()]
print(f"{len(jobs)} frames total ({sum(len(d['reasoning_image']) for d in data)} future), {len(todo)} to do", flush=True)

sam = sam_model_registry["vit_h"](checkpoint=a.ckpt).cuda().eval()
gen = SamAutomaticMaskGenerator(sam, points_per_batch=a.points_per_batch)
t0 = time.time()
for i, (sid, fdir, k) in enumerate(todo, 1):
    img = np.array(Image.open(os.path.join(fdir, f"f{k}.jpg")).convert("RGB").resize((256, 256)))
    with torch.no_grad():
        masks = gen.generate(img)
    masks = sorted(masks, key=lambda m: m["predicted_iou"] * m["stability_score"], reverse=True)[:8]
    arr = np.stack([m["segmentation"] for m in masks]) if masks else np.zeros((0, 256, 256), bool)
    score = np.array([m["predicted_iou"] * m["stability_score"] for m in masks], np.float32)
    (out / sid).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / sid / f"f{k}.npz", masks=np.packbits(arr, axis=-1), n=len(masks), score=score)
    if i % 100 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"{i}/{len(todo)}  {el/i:.2f} s/frame  ETA {(len(todo)-i)*el/i/3600:.1f} h", flush=True)
print("done")
