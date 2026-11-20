"""Option A of the VLM ladder: frozen Qwen3-VL features that keep WHERE things are in the frame.
Same segments, frames and prompt as 03_features.py, but instead of averaging all of a frame's image tokens into one
vector, the frame's token map (about 8 x 14 tokens for 448 x 252) is average-pooled to a G x G grid (default 2 x 2),
so small parts in one corner are not washed out by the rest of the table.
Saved in chunks of 2,048 segments (resumable): <out>/<split>_<k>.pt with
  frames [n, 12, G*G, nL, 4096] fp16  (slots 0-7 history, oldest first, right-aligned; 8-11 recent; cells row-major)
  mask   [n, 12] bool, last [n, nL, 4096] fp16, y [n, 3], ids, layers, grid
usage: python 05_features_grid.py /workspace/data/a101_vlm /workspace/outputs/q3_grid [--grid 2] [--layers 27] [--limit 64]
"""
import argparse, json, math, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layers", default="27")
ap.add_argument("--grid", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LAYERS = [int(x) for x in a.layers.split(",")]
G = a.grid

proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
IMG = model.config.image_token_id
MERGE = model.config.vision_config.spatial_merge_size
img = lambda q, f: Image.open(os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")).convert("RGB")   # noqa: E731


def message(q):
    content = []
    if q["hist"]:
        content.append({"type": "text", "text": "Earlier frames from the last minute, oldest first:"})
        content += [{"type": "image", "image": img(q, f)} for f in q["hist"]]
    content.append({"type": "text", "text": "The last 2 seconds, oldest first:"})
    content += [{"type": "image", "image": img(q, f)} for f in q["recent"]]
    content.append({"type": "text", "text": "The person is assembling or disassembling a toy vehicle. What will they do next?"})
    return [{"role": "user", "content": content}]


def runs(pos):
    out, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            out.append((pos[start], pos[i - 1] + 1)); start = i
    return out


first = True
for split in ("train", "validation"):
    data = json.load(open(os.path.join(a.vlm, f"{split}.json")))[: a.limit or None]
    nchunks = math.ceil(len(data) / a.chunk)
    t0, seen = time.time(), 0
    for k in range(nchunks):
        path = os.path.join(a.out, f"{split}_{k}.pt")
        if os.path.exists(path):
            continue
        part = data[k * a.chunk:(k + 1) * a.chunk]
        Fr = torch.zeros(len(part), 12, G * G, len(LAYERS), 4096, dtype=torch.float16)
        M = torch.zeros(len(part), 12, dtype=torch.bool)
        L = torch.zeros(len(part), len(LAYERS), 4096, dtype=torch.float16)
        for s in range(0, len(part), a.bs):
            batch = part[s:s + a.bs]
            inp = proc.apply_chat_template([message(q) for q in batch], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True).hidden_states
            H = torch.stack([hs[l] for l in LAYERS], 2)                      # [B, T, nL, D]
            thw = inp["image_grid_thw"].tolist()                             # one row per image, batch order
            im = 0
            for b, q in enumerate(batch):
                rr = runs((inp["input_ids"][b] == IMG).nonzero().flatten().tolist())
                nh = len(q["hist"])
                assert len(rr) == nh + 4, (len(rr), nh)
                slots = list(range(8 - nh, 8)) + [8, 9, 10, 11]
                for slot, (i0, i1) in zip(slots, rr):
                    t, h, w = thw[im]; im += 1
                    h, w = h // MERGE, w // MERGE
                    assert i1 - i0 == t * h * w, (i1 - i0, t, h, w)
                    x = H[b, i0:i1].float().reshape(t * h, w, -1).permute(2, 0, 1)   # [nL*D, h, w]
                    x = F.adaptive_avg_pool2d(x, G).reshape(len(LAYERS), 4096, G * G).permute(2, 0, 1)
                    Fr[s + b, slot] = x.cpu().half()
                    M[s + b, slot] = True
                    if first:
                        print(f"token map per frame: {h} x {w} -> pooled to {G} x {G}", flush=True); first = False
                L[s + b] = H[b, -1].cpu()
            assert im == len(thw)
            seen += len(batch)
            if (s // a.bs) % 50 == 0:
                el = time.time() - t0
                left = len(data) - (k * a.chunk + s + len(batch))
                print(f"{split} chunk {k + 1}/{nchunks} | {seen / el:.1f} segments/s, ~{left / max(seen / el, 1e-6) / 60:.0f} min left in {split}", flush=True)
        torch.save({"frames": Fr, "mask": M, "last": L, "y": torch.tensor([q["y"] for q in part]),
                    "ids": [q["id"] for q in part], "layers": LAYERS, "grid": G}, path)
        print(f"saved {path} ({len(part)} segments)", flush=True)
print("done")
