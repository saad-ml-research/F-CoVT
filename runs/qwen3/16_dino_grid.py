"""Plain-ViT frame features (no language model): frozen DINOv2 on the SAME frames as the VLM banks (a101_vlm json),
pooled to the same 3 x 3 grid, so 04_probe.py can swap the VLM frame tokens for ViT tokens (rows d_*, --dino).
Frames are resized to 252 x 448 (no crop, so the whole table stays in view) = 18 x 32 patches of 14 px, ImageNet-normalized;
the patch map is average-pooled to G x G cells (row-major) and the CLS token is kept as well.
By default only the 4 recent frames (last 2 s); --hist also does the 8 history frames (last 60 s).
Saves <out>/dino.pt = {split: {"rec": [n, 4, G*G, D] fp16, "rec_cls": [n, 4, D] fp16, ("hist", "hist_cls", "hist_mask"),
"y": [n, 3], "ids"}}, segment order as the VLM bank.
usage: python 16_dino_grid.py /workspace/data/a101_vlm /workspace/outputs/dino [--model facebook/dinov2-large] [--limit 64]
"""
import argparse, json, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("out")
ap.add_argument("--model", default="facebook/dinov2-large")
ap.add_argument("--grid", type=int, default=3)
ap.add_argument("--hist", action="store_true")
ap.add_argument("--bs", type=int, default=64, help="segments per batch")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
G, H, W = a.grid, 252, 448
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
D = model.config.hidden_size
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
pool = ThreadPoolExecutor(16)


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC))


@torch.no_grad()
def encode(paths):
    """list of image paths -> grid [n, G*G, D], cls [n, D]."""
    x = torch.from_numpy(np.stack(list(pool.map(load, paths)))).cuda().permute(0, 3, 1, 2).float() / 255
    x = ((x - MEAN) / STD).to(torch.bfloat16)
    out = []
    for i in range(0, len(x), 256):
        h = model(pixel_values=x[i:i + 256]).last_hidden_state.float()          # [b, 1 + regs + 18*32, D]
        cls, p = h[:, 0], h[:, -(H // 14) * (W // 14):]
        p = p.reshape(len(p), H // 14, W // 14, D).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(p, G).flatten(2).transpose(1, 2)               # [b, G*G, D], row-major
        out.append((g.half().cpu(), cls.half().cpu()))
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


img = lambda q, f: os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")   # noqa: E731
res, t0 = {}, time.time()
for split in ("train", "validation"):
    data = json.load(open(os.path.join(a.vlm, f"{split}.json")))[: a.limit or None]
    n = len(data)
    R = torch.zeros(n, 4, G * G, D, dtype=torch.float16); RC = torch.zeros(n, 4, D, dtype=torch.float16)
    if a.hist:
        Hh = torch.zeros(n, 8, G * G, D, dtype=torch.float16); HC = torch.zeros(n, 8, D, dtype=torch.float16)
        HM = torch.zeros(n, 8, dtype=torch.bool)
    for s in range(0, n, a.bs):
        batch = data[s:s + a.bs]
        assert all(len(q["recent"]) == 4 for q in batch)
        g, c = encode([img(q, f) for q in batch for f in q["recent"]])
        R[s:s + len(batch)] = g.view(len(batch), 4, G * G, D); RC[s:s + len(batch)] = c.view(len(batch), 4, D)
        if a.hist:
            paths, slots = [], []
            for b, q in enumerate(batch):
                nh = len(q["hist"])
                for j, f in enumerate(q["hist"]):
                    paths.append(img(q, f)); slots.append((s + b, 8 - nh + j))          # right-aligned, oldest first
            if paths:
                g, c = encode(paths)
                for k, (i, j) in enumerate(slots):
                    Hh[i, j], HC[i, j], HM[i, j] = g[k], c[k], True
        if (s // a.bs) % 50 == 0:
            done = s + len(batch); el = time.time() - t0
            print(f"{split}: {done}/{n} segments | {(time.time() - t0) / 60:.1f} min | ~{(n - done) * el / max(done, 1) / 60:.0f} min left in {split}", flush=True)
    res[split] = {"rec": R, "rec_cls": RC, "y": torch.tensor([q["y"] for q in data]), "ids": [q["id"] for q in data]}
    if a.hist:
        res[split].update({"hist": Hh, "hist_cls": HC, "hist_mask": HM})
    t0 = time.time()
torch.save(res, os.path.join(a.out, "dino.pt"))
print(f"saved {os.path.join(a.out, 'dino.pt')} | model {a.model}, {D}-d, grid {G} x {G}, history frames: {a.hist}")
