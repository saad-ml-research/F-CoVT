"""Text history, style sweep: the frozen Qwen3-VL language model reads the PREDICTED past actions (06_recog.py) as text
in several styles; each past step -> mean of its words' layer-27 states (+ the final state of the whole text, "last").
Styles (last K steps, oldest first):
  list    "Steps so far, oldest first: pick up screw; unscrew first wheel."
  narr    "So far the person has, in order: first picked: pick up screw, then: unscrew first wheel."
  timed   "Steps so far: 1. (12 s ago) pick up screw 2. (4 s ago) unscrew first wheel."
  conf    "Steps so far (recognizer confidence): pick up screw (60%); unscrew first wheel (35%)."
  top3    "Steps so far (best guesses): pick up screw or pick up nut or pick up wheel; ..."
  struct  "Steps so far: verb pick up, object screw; verb unscrew, object first wheel."
To save disk, vectors are compressed 4096 -> --dim with PCA fit on train. Saves <out>/text_<style>.pt =
{split: {"pred": [n, K, dim] fp16, "last": [n, 1, dim] fp16, "mask": [n, K]}}.
usage: python 12_text_styles.py CSVs recog.pt OUT [--styles list,narr,timed,conf,top3,struct] [--K 8] [--limit 64]
"""
import argparse, csv, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("csvs"); ap.add_argument("recog"); ap.add_argument("out")
ap.add_argument("--styles", default="list,narr,timed,conf,top3,struct")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dim", type=int, default=1024)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

VN, rows = {}, {"train": [], "validation": []}       # action id -> (verb, object); per split: (start, end) in file order
with open(os.path.join(a.csvs, "train.csv")) as f:
    for r in csv.DictReader(f):
        VN[int(r["action_id"])] = (r["verb_cls"].strip().replace("_", " "), r["noun_cls"].strip().replace("_", " "))
        if r["video"].endswith("/C10119_rgb.mp4"):
            rows["train"].append((int(r["start_frame"]), int(r["end_frame"])))
with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
    for r in csv.DictReader(f):
        VN[int(r["action"])] = (r["verb_cls"].strip().replace("_", " "), r["noun_cls"].strip().replace("_", " "))
        rows["validation"].append((int(r["start"]), int(r["end"])))
R = torch.load(a.recog)
NA = R["sizes"][0]
name = lambda c: f"{VN[c][0]} {VN[c][1]}" if c in VN else "unknown action"   # noqa: E731
proc = AutoProcessor.from_pretrained(a.model)
tok = proc.tokenizer; tok.padding_side = "right"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()


def render(style, steps):
    """steps: list of dicts (name, ago, conf, top3, verb, obj) -> text, char spans of each step."""
    heads = {"list": "Steps so far, oldest first: ", "narr": "So far the person has done, in order: ",
             "timed": "Steps so far: ", "conf": "Steps so far (recognizer confidence): ",
             "top3": "Steps so far (best guesses, oldest first): ", "struct": "Steps so far, oldest first: "}
    t, sp = heads[style], []
    for i, s in enumerate(steps):
        if style == "list":
            piece, sep = s["name"], "; "
        elif style == "narr":
            piece, sep = ("first " if i == 0 else "then ") + s["name"], ", "
        elif style == "timed":
            piece, sep = f"{i + 1}. ({s['ago']:.0f} s ago) {s['name']}", " "
        elif style == "conf":
            piece, sep = f"{s['name']} ({100 * s['conf']:.0f}%)", "; "
        elif style == "top3":
            piece, sep = " or ".join(s["top3"]), "; "
        else:
            piece, sep = f"verb {s['verb']}, object {s['obj']}", "; "
        sp.append((len(t), len(t) + len(piece))); t += piece + (sep if i < len(steps) - 1 else ".")
    return t + " What will the person do next?", sp


@torch.no_grad()
def encode(texts, spans):
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")
    H = model.model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
                    output_hidden_states=True).hidden_states[a.layer].float()
    out = torch.zeros(len(texts), a.K, 4096); mask = torch.zeros(len(texts), a.K, dtype=torch.bool)
    last = H[torch.arange(len(texts)), enc["attention_mask"].sum(1).cuda() - 1].cpu()
    for b, sp in enumerate(spans):
        o = off[b]
        for i, (c0, c1) in enumerate(sp):
            sel = ((o[:, 0] < c1) & (o[:, 1] > c0) & (o[:, 1] > o[:, 0])).nonzero().flatten()
            slot = a.K - len(sp) + i
            out[b, slot] = H[b, sel].mean(0).cpu(); mask[b, slot] = True
    return out, last, mask


t0 = time.time()
for style in a.styles.split(","):
    full = {}
    for s in ("train", "validation"):
        Hidx = R[s]["hist"][:, -a.K:]
        pred, prob = R[s]["pred"][:, 0], R[s]["prob"][:, :NA].float()
        n = a.limit or len(Hidx)
        X = torch.zeros(n, a.K, 4096, dtype=torch.float16); L = torch.zeros(n, 1, 4096, dtype=torch.float16)
        M = torch.zeros(n, a.K, dtype=torch.bool)
        for b0 in range(0, n, a.bs):
            texts, spans = [], []
            for i in range(b0, min(n, b0 + a.bs)):
                cut = rows[s][i][0] - 30
                steps = [{"name": name(int(pred[j])), "ago": max(0, cut - rows[s][j][1]) / 30, "conf": float(prob[j, pred[j]]),
                          "top3": [name(int(c)) for c in prob[j].topk(3).indices], "verb": VN.get(int(pred[j]), ("?", "?"))[0],
                          "obj": VN.get(int(pred[j]), ("?", "?"))[1]} for j in Hidx[i].tolist() if j >= 0]
                if not steps:
                    steps = [{"name": "nothing yet", "ago": 0, "conf": 1.0, "top3": ["nothing yet"], "verb": "none", "obj": "none"}]
                t, sp = render(style, steps); texts.append(t); spans.append(sp)
            x, l, m = encode(texts, spans)
            X[b0:b0 + len(texts)], L[b0:b0 + len(texts), 0], M[b0:b0 + len(texts)] = x.half(), l.half(), m
        full[s] = (X, L, M)
        if a.limit:
            print(f"[{style}] example: {texts[0][:300]}", flush=True)
        print(f"[{style}] {s}: {n} segments | {(time.time() - t0) / 60:.1f} min", flush=True)
    # PCA on train step vectors (+ last states), applied to both splits
    Xtr, Ltr, Mtr = full["train"]
    sample = torch.cat([Xtr[Mtr][:30000].float(), Ltr[:10000, 0].float()]).cuda()
    mu = sample.mean(0)
    _, _, V = torch.pca_lowrank(sample - mu, q=a.dim, center=False)
    res = {}
    for s, (X, L, M) in full.items():
        proj = lambda z: ((z.float().cuda() - mu) @ V).half().cpu()   # noqa: E731
        res[s] = {"pred": torch.cat([proj(X[i:i + 4096]) for i in range(0, len(X), 4096)]),
                  "last": torch.cat([proj(L[i:i + 4096]) for i in range(0, len(L), 4096)]), "mask": M}
    torch.save(res, os.path.join(a.out, f"text_{style}.pt"))
    print(f"[{style}] saved text_{style}.pt ({a.dim}-d) | {(time.time() - t0) / 60:.1f} min", flush=True)
    del full, res
print("done")
