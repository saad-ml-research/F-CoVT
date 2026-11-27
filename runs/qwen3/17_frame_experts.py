"""Vision experts on ONE frame: the last frame before the cut (0 s; the action starts 1 s later), camera C10119,
the same 448 x 252 RGB frame the VLM sees (a101_vlm json, recent[3]). Each expert is pooled to several token budgets
so 04_probe.py can test 1 / 4 / 9 / 16 tokens (hands: 2 = one per hand).
  dino   DINOv2-Large, 252 x 448 input (18 x 32 patches), no crop. g1 = CLS token; g4 / g9 / g16 = patch map
         average-pooled to 2 x 2 / 3 x 3 / 4 x 4 cells (row-major). 1024-d.
  depth  Depth Anything V2 Large (relative depth), processor defaults; depth map divided by its median (per image),
         then per cell [mean, std, 5th percentile, 95th percentile] -> 4-d tokens, g1 / g4 / g9 / g16.
  sam    SAM 3 (facebook/sam3) at its 1008 px input (the frame is upsampled): the ViT backbone tokens (72 x 72, 1024-d),
         average-pooled to g1 / g4 / g9 / g16.
  hands  MediaPipe Hands on the frame upscaled 2x (single image, this camera only): per hand slot (by handedness)
         [present, score, wrist x, wrist y (image-normalized), 20 joints relative to the wrist (x, y, z)] = 64-d;
         g2 = 2 tokens. Missing hand = zeros.
Saves <out>/<expert>.pt = {split: {"g1"/"g4"/"g9"/"g16" (or "g2"): [n, k, D] fp16, "y": [n, 3], "ids"}}, segment
order as the VLM bank. Resumable: an expert whose file exists is skipped.
usage: python 17_frame_experts.py /workspace/data/a101_vlm /workspace/outputs/experts [--experts dino,depth,sam,hands] [--limit 64]
"""
import argparse, json, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("out")
ap.add_argument("--experts", default="dino,depth,sam,hands")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
GRIDS = (1, 2, 3, 4)
pool_io = ThreadPoolExecutor(16)
data = {s: json.load(open(os.path.join(a.vlm, f"{s}.json")))[: a.limit or None] for s in ("train", "validation")}
path = lambda q: os.path.join(a.vlm, "frames", q["seq"], f"{q['recent'][3]:06d}.jpg")   # noqa: E731  frame at 0 s
rgb = lambda p: Image.open(p).convert("RGB")   # noqa: E731


def grids(fmap):
    """[B, C, H, W] -> {"g1".."g16": [B, k, C]} average-pooled."""
    return {f"g{g * g}": F.adaptive_avg_pool2d(fmap, g).flatten(2).transpose(1, 2) for g in GRIDS}


def run(name, encode, keys):
    dst = os.path.join(a.out, f"{name}.pt")
    if os.path.exists(dst):
        print(f"[{name}] exists, skipped", flush=True); return
    t0, res = time.time(), {}
    for s, rows in data.items():
        parts = {k: [] for k in keys}
        for b in range(0, len(rows), a.bs):
            imgs = list(pool_io.map(lambda q: rgb(path(q)), rows[b:b + a.bs]))
            out = encode(imgs)
            for k in keys:
                parts[k].append(out[k].float().half().cpu())
            if (b // a.bs) % 200 == 0:
                done = b + len(imgs); el = time.time() - t0
                print(f"[{name}] {s}: {done}/{len(rows)} | {el / 60:.1f} min", flush=True)
        res[s] = {k: torch.cat(v) for k, v in parts.items()}
        res[s]["y"] = torch.tensor([q["y"] for q in rows]); res[s]["ids"] = [q["id"] for q in rows]
        for k in keys:
            x = res[s][k].float()
            print(f"[{name}] {s} {k}: {tuple(x.shape)} | mean {x.mean():.3f} sd {x.std():.3f} | non-zero rows {(x.abs().sum(-1) > 0).float().mean():.2f}", flush=True)
    torch.save(res, dst)
    print(f"[{name}] saved {dst} | {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ experts
def make_dino():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.bfloat16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def enc(imgs):
        x = torch.from_numpy(np.stack([np.asarray(i.resize((448, 252), Image.BICUBIC)) for i in imgs])).cuda()
        x = ((x.permute(0, 3, 1, 2).float() / 255 - mean) / std).to(torch.bfloat16)
        h = m(pixel_values=x).last_hidden_state.float()
        p = h[:, -18 * 32:].reshape(len(x), 18, 32, -1).permute(0, 3, 1, 2)
        out = grids(p); out["g1"] = h[:, :1]                                    # g1 = CLS
        return out
    return enc


def make_depth():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Large-hf"
    proc = AutoImageProcessor.from_pretrained(name)
    m = AutoModelForDepthEstimation.from_pretrained(name, dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def enc(imgs):
        inp = proc(images=imgs, return_tensors="pt").to("cuda")
        d = m(pixel_values=inp["pixel_values"].half()).predicted_depth.float()       # [B, H, W] relative (inverse) depth
        d = d / d.flatten(1).median(1).values.clamp(min=1e-6)[:, None, None]
        d = d[:, None]
        out = {}
        for g in GRIDS:
            mu = F.adaptive_avg_pool2d(d, g); sd = (F.adaptive_avg_pool2d(d * d, g) - mu * mu).clamp(min=0).sqrt()
            B, _, H, W = d.shape
            cells = d.unfold(2, H // g, H // g).unfold(3, W // g, W // g)[:, 0, :g, :g].reshape(B, g, g, -1)
            lo, hi = cells.quantile(0.05, dim=-1), cells.quantile(0.95, dim=-1)
            out[f"g{g * g}"] = torch.stack([mu[:, 0], sd[:, 0], lo, hi], -1).reshape(B, g * g, 4)
        return out
    return enc


def make_sam():
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    m = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).cuda().eval()
    first = [True]

    def coarsest(v):
        """the lowest-resolution [B, C, H, W] map in whatever get_vision_features returns (or a [B, N, C] grid)."""
        found = []
        def walk(o):
            if torch.is_tensor(o):
                found.append(o)
            elif isinstance(o, (list, tuple)):
                for x in o: walk(x)
            elif isinstance(o, dict):
                for x in o.values(): walk(x)
            elif hasattr(o, "to_tuple"):
                walk(o.to_tuple())
        walk(v)
        maps = [t for t in found if t.dim() == 4 and min(t.shape[-2:]) >= 4]
        if first[0]:
            print("[sam] vision outputs:", [tuple(t.shape) for t in found], flush=True); first[0] = False
        vit = [t for t in found if t.dim() == 3 and int(round(t.shape[1] ** 0.5)) ** 2 == t.shape[1]]
        if vit:                                       # prefer the ViT backbone tokens (72 x 72 x 1024) over the 256-d FPN neck
            t = vit[0]; s = int(round(t.shape[1] ** 0.5))
            return t.transpose(1, 2).reshape(len(t), -1, s, s)
        if maps:
            return min(maps, key=lambda t: t.shape[-1] * t.shape[-2])
        t = [t for t in found if t.dim() == 3][-1]                                # [B, N, C] -> square map
        s = int(round(t.shape[1] ** 0.5)); return t[:, -s * s:].transpose(1, 2).reshape(len(t), -1, s, s)

    @torch.no_grad()
    def enc(imgs):
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        return grids(coarsest(m.get_vision_features(pixel_values=px)).float())
    return enc


def make_hands():
    import mediapipe as mp
    det = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, model_complexity=1)

    def one(img):
        x = np.asarray(img.resize((img.width * 2, img.height * 2), Image.BICUBIC))
        r = det.process(x)
        tok = np.zeros((2, 64), np.float32)
        if r.multi_hand_landmarks:
            for lm, hd in zip(r.multi_hand_landmarks, r.multi_handedness):
                c = hd.classification[0]; slot = 0 if c.label == "Left" else 1
                if tok[slot, 0] and tok[slot, 1] >= c.score:
                    continue
                j = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)        # 21 x 3, x/y image-normalized
                rel = (j[1:] - j[:1]) * np.array([10, 10, 10], np.float32)            # hands are small: scale up
                tok[slot] = np.concatenate([[1.0, c.score], j[0, :2], rel.ravel()])
        return tok

    def enc(imgs):
        return {"g2": torch.from_numpy(np.stack([one(i) for i in imgs]))}
    return enc


MAKERS = {"dino": (make_dino, ["g1", "g4", "g9", "g16"]), "depth": (make_depth, ["g1", "g4", "g9", "g16"]),
          "sam": (make_sam, ["g1", "g4", "g9", "g16"]), "hands": (make_hands, ["g2"])}
for e in a.experts.split(","):
    mk, keys = MAKERS[e]
    print(f"[{e}] loading", flush=True)
    run(e, mk(), keys)
    torch.cuda.empty_cache()
print("done")
