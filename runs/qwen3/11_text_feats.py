"""Text history, step A: the frozen Qwen3-VL language model reads the past actions as TEXT (no images), and each
past step becomes one vector = the mean of its words' layer-27 states (in the context of the whole sequence).
Two versions per segment: the TRUE past labels (annotations) and the PREDICTED ones (06_recog.py top-1), last K steps.
  "Steps so far, oldest first: pick up wheel; attach wheel; ...; screw wheel. What will the person do next?"
Saves <out>/text.pt = {split: {"true": [n, K, 4096] fp16, "pred": [n, K, 4096] fp16, "mask": [n, K]}}, steps right-aligned
like the history indices of recog.pt, so 04_probe.py can use them as history tokens (+hX / +hXP).
usage: python 11_text_feats.py /workspace/data/a101_anticip/data/CSVs /workspace/outputs/recog256v/recog.pt
                               /workspace/outputs/q3_text [--K 8] [--limit 64]
"""
import argparse, csv, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("csvs"); ap.add_argument("recog"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

name = {}                                                      # action id -> "verb object"
for fn, ids in (("train.csv", ("action_id", "verb_cls", "noun_cls")), ("validation_challenge.csv", ("action", "verb_cls", "noun_cls"))):
    with open(os.path.join(a.csvs, fn)) as f:
        for r in csv.DictReader(f):
            name[int(r[ids[0]])] = f"{r[ids[1]].strip()} {r[ids[2]].strip()}".replace("_", " ")
R = torch.load(a.recog)
proc = AutoProcessor.from_pretrained(a.model)
tok = proc.tokenizer
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
print(f"{len(name)} action names, e.g. {list(name.values())[:3]}", flush=True)


def encode(texts_steps):
    """texts_steps: list of lists of step strings -> [B, K, 4096] mean state per step (right-aligned), mask."""
    texts, spans = [], []
    for steps in texts_steps:
        t, sp = "Steps so far, oldest first: ", []
        for i, st in enumerate(steps):
            sp.append((len(t), len(t) + len(st))); t += st + ("; " if i < len(steps) - 1 else ".")
        t += " What will the person do next?"
        texts.append(t); spans.append(sp)
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")
    with torch.no_grad():
        H = model.model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
                        output_hidden_states=True).hidden_states[a.layer].float()
    out = torch.zeros(len(texts), a.K, 4096); mask = torch.zeros(len(texts), a.K, dtype=torch.bool)
    for b, sp in enumerate(spans):
        o = off[b]
        for i, (c0, c1) in enumerate(sp):
            sel = ((o[:, 0] < c1) & (o[:, 1] > c0) & (o[:, 1] > o[:, 0])).nonzero().flatten()
            slot = a.K - len(sp) + i
            out[b, slot] = H[b, sel].mean(0).cpu(); mask[b, slot] = True
    return out.half(), mask


tok.padding_side = "right"
res = {}
t0 = time.time()
for s in ("train", "validation"):
    Hidx = R[s]["hist"][:, -a.K:]
    n = len(Hidx) if not a.limit else a.limit
    res[s] = {"true": torch.zeros(n, a.K, 4096, dtype=torch.float16), "pred": torch.zeros(n, a.K, 4096, dtype=torch.float16),
              "mask": torch.zeros(n, a.K, dtype=torch.bool)}
    for src, lab in (("true", R[s]["y"][:, 0]), ("pred", R[s]["pred"][:, 0])):
        for b0 in range(0, n, a.bs):
            steps = [[name[int(lab[j])] for j in Hidx[i] if j >= 0] or ["nothing yet"] for i in range(b0, min(n, b0 + a.bs))]
            x, m = encode(steps)
            res[s][src][b0:b0 + len(steps)] = x
            if src == "true":
                res[s]["mask"][b0:b0 + len(steps)] = m
        print(f"{s} {src}: {n} segments | {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.limit:
        print("example text steps:", [name[int(R[s]['y'][j, 0])] for j in Hidx[min(5, n - 1)] if j >= 0], flush=True)
torch.save(res, os.path.join(a.out, "text.pt"))
print(f"saved {os.path.join(a.out, 'text.pt')} | done in {(time.time() - t0) / 60:.1f} min")
