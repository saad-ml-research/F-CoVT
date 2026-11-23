"""Option 5, step 1: frozen Qwen3-VL features of EVERY annotated action segment (camera C10119), for recognizing past
actions. Uses the frames already extracted by 01_frames.py: per segment, 4 frames spread over those inside its
annotated [start, end] (89% of segments have 2+ inside, 65% have 4+); if fewer than 4 are inside, the nearest frames
to the segment's middle fill up. Per frame: layer-27 image tokens pooled to a 3 x 3 grid (as 05_features_grid.py).
Segment order = train.csv rows of this camera, then validation_challenge.csv (same as runs/qwen3/06_recog.py).
Saves chunks of 2,048 (resumable): <out>/<split>_<k>.pt = {frames [n, 4, 9, 4096] fp16, inside [n] (frames inside), y [n, 3]}
usage: python 09_seg_feats.py /workspace/data/a101_anticip/data/CSVs /workspace/data/a101_vlm /workspace/outputs/q3_seg [--limit 64]
"""
import argparse, bisect, csv, math, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("csvs"); ap.add_argument("vlm"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--grid", type=int, default=3)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
G = a.grid


def rows_of(split):
    out = []
    if split == "train":
        with open(os.path.join(a.csvs, "train.csv")) as f:
            for r in csv.DictReader(f):
                if r["video"].endswith("/C10119_rgb.mp4"):
                    out.append((r["video"].split("/")[0], int(r["start_frame"]), int(r["end_frame"]),
                                [int(r["action_id"]), int(r["verb_id"]), int(r["noun_id"])]))
    else:
        with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
            for r in csv.DictReader(f):
                out.append((r["video"].split("/")[0], int(r["start"]), int(r["end"]), [int(r["action"]), int(r["verb"]), int(r["noun"])]))
    return out


FR = {}


def frames_of(seq):
    if seq not in FR:
        d = os.path.join(a.vlm, "frames", seq)
        FR[seq] = sorted(int(n[:-4]) for n in os.listdir(d) if n.endswith(".jpg"))
    return FR[seq]


def pick(seq, st, en):
    fr = frames_of(seq)
    inside = fr[bisect.bisect_left(fr, st): bisect.bisect_right(fr, en)]
    if len(inside) >= 4:
        sel = [inside[round(i * (len(inside) - 1) / 3)] for i in range(4)]
    else:
        mid = (st + en) / 2
        extra = sorted((f for f in fr if f not in inside), key=lambda f: abs(f - mid))[: 4 - len(inside)]
        sel = sorted(inside + extra)
    return sel, len(inside)


proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
IMG, MERGE = model.config.image_token_id, model.config.vision_config.spatial_merge_size


def message(seq, sel):
    content = [{"type": "text", "text": "Frames of one short action, in order:"}]
    content += [{"type": "image", "image": Image.open(os.path.join(a.vlm, "frames", seq, f"{f:06d}.jpg")).convert("RGB")} for f in sel]
    content.append({"type": "text", "text": "The person is assembling or disassembling a toy vehicle. Which action are they doing?"})
    return [{"role": "user", "content": content}]


def runs(pos):
    out, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            out.append((pos[start], pos[i - 1] + 1)); start = i
    return out


for split in ("train", "validation"):
    data = rows_of(split)[: a.limit or None]
    nchunks = math.ceil(len(data) / a.chunk)
    t0, seen = time.time(), 0
    for k in range(nchunks):
        path = os.path.join(a.out, f"{split}_{k}.pt")
        if os.path.exists(path):
            continue
        part = data[k * a.chunk:(k + 1) * a.chunk]
        Fr = torch.zeros(len(part), 4, G * G, 4096, dtype=torch.float16)
        INS = torch.zeros(len(part), dtype=torch.long)
        for s in range(0, len(part), a.bs):
            batch = part[s:s + a.bs]
            sels = [pick(seq, st, en) for seq, st, en, _ in batch]
            inp = proc.apply_chat_template([message(b[0], sel) for b, (sel, _) in zip(batch, sels)], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                H = model(**inp, output_hidden_states=True).hidden_states[a.layer]
            thw = inp["image_grid_thw"].tolist(); im = 0
            for b in range(len(batch)):
                rr = runs((inp["input_ids"][b] == IMG).nonzero().flatten().tolist())
                assert len(rr) == 4, len(rr)
                for slot, (i0, i1) in enumerate(rr):
                    t, h, w = thw[im]; im += 1
                    h, w = h // MERGE, w // MERGE
                    x = H[b, i0:i1].float().reshape(t * h, w, -1).permute(2, 0, 1)
                    Fr[s + b, slot] = F.adaptive_avg_pool2d(x, G).flatten(1).t().cpu().half()
                INS[s + b] = sels[b][1]
            seen += len(batch)
            if (s // a.bs) % 25 == 0:
                el = time.time() - t0
                left = len(data) - (k * a.chunk + s + len(batch))
                print(f"{split} chunk {k + 1}/{nchunks} | {seen / el:.1f} segments/s, ~{left / max(seen / el, 1e-6) / 60:.0f} min left in {split}", flush=True)
        torch.save({"frames": Fr, "inside": INS, "y": torch.tensor([r[3] for r in part]), "layer": a.layer, "grid": G}, path)
        print(f"saved {path} ({len(part)} segments)", flush=True)
print("done")
