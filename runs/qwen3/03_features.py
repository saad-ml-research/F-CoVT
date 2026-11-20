"""Rung 1 of the VLM ladder: frozen Qwen3-VL features for the official anticipation segments (camera C10119).
Per segment the model reads the 8 history frames (last 60 s; fewer near the video start) and the 4 recent frames
(last 2 s) with a short question. For every frame we keep the MEAN of that image's tokens at three depths of the
language model (layers 18, 27, 36 of 36), and the state of the last prompt token at the same depths.
Saved in chunks of 2,048 segments (resumable): <out>/<split>_<k>.pt with
  frames [n, 12, 3, 4096] fp16  (slots 0-7 history, oldest first, right-aligned; 8-11 recent)
  mask   [n, 12] bool            (False = no frame: before the video start)
  last   [n, 3, 4096] fp16, y [n, 3], ids
usage: python 03_features.py /workspace/data/a101_vlm /workspace/outputs/q3_feats [--limit 64] [--bs 8]
"""
import argparse, json, math, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layers", default="18,27,36")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LAYERS = [int(x) for x in a.layers.split(",")]

proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
IMG = model.config.image_token_id
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
    """split sorted token positions into runs of consecutive positions (one run per image)."""
    out, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            out.append((pos[start], pos[i - 1] + 1)); start = i
    return out


for split in ("train", "validation"):
    data = json.load(open(os.path.join(a.vlm, f"{split}.json")))[: a.limit or None]
    nchunks = math.ceil(len(data) / a.chunk)
    t0, seen = time.time(), 0
    for k in range(nchunks):
        path = os.path.join(a.out, f"{split}_{k}.pt")
        if os.path.exists(path):
            continue
        part = data[k * a.chunk:(k + 1) * a.chunk]
        F = torch.zeros(len(part), 12, len(LAYERS), 4096, dtype=torch.float16)
        M = torch.zeros(len(part), 12, dtype=torch.bool)
        L = torch.zeros(len(part), len(LAYERS), 4096, dtype=torch.float16)
        for s in range(0, len(part), a.bs):
            batch = part[s:s + a.bs]
            inp = proc.apply_chat_template([message(q) for q in batch], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True).hidden_states
            H = torch.stack([hs[l] for l in LAYERS], 2)                      # [B, T, nL, D]
            for b, q in enumerate(batch):
                pos = (inp["input_ids"][b] == IMG).nonzero().flatten().tolist()
                rr = runs(pos)
                nh = len(q["hist"])
                assert len(rr) == nh + 4, (len(rr), nh)
                slots = list(range(8 - nh, 8)) + [8, 9, 10, 11]
                for slot, (i0, i1) in zip(slots, rr):
                    F[s + b, slot] = H[b, i0:i1].float().mean(0).cpu().half()
                    M[s + b, slot] = True
                L[s + b] = H[b, -1].cpu()
            seen += len(batch)
            if (s // a.bs) % 50 == 0:
                el = time.time() - t0
                left = len(data) - (k * a.chunk + s + len(batch))
                print(f"{split} chunk {k + 1}/{nchunks} | {seen / el:.1f} segments/s, ~{left / max(seen / el, 1e-6) / 60:.0f} min left in {split}", flush=True)
        torch.save({"frames": F, "mask": M, "last": L, "y": torch.tensor([q["y"] for q in part]),
                    "ids": [q["id"] for q in part], "layers": LAYERS}, path)
        print(f"saved {path} ({len(part)} segments)", flush=True)
print("done")
