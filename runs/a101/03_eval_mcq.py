"""Evaluate a (merged) Qwen2.5-VL model on the Assembly101 next-action questions.
The prompt is built exactly like training (Future-L1 loader, CONTEXT=4): system "You are a helpful assistant.",
"Earlier frames (oldest first):" f1-f4, "Current frame:" "frame_1" f5, then the question; stills <= 256*28*28 px.
PREFIX="..." is appended after the assistant start (e.g. a forced thought sentence); otherwise the model writes freely.
Writes <out>/answers.jsonl (per question) and prints accuracy overall and by verb.
usage: python 03_eval_mcq.py MODEL /workspace/data/a101/validation.json OUT_DIR [--limit 200]
"""
import argparse, collections, json, os, re
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--context", type=int, default=4)
ap.add_argument("--max_new_tokens", type=int, default=160)
a = ap.parse_args()
PREFIX = os.environ.get("FCOVT_PREFIX", "")
MAXP, MINP = 256 * 28 * 28, 64 * 28 * 28
TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S); LET = re.compile(r"\b([ABCDE])\b")

proc = AutoProcessor.from_pretrained(a.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                           attn_implementation="flash_attention_2", device_map="cuda").eval()
data = json.load(open(a.data))[: a.limit or None]
os.makedirs(a.out, exist_ok=True)
fout = open(os.path.join(a.out, "answers.jsonl"), "w")
right, by_verb, n_think = 0, collections.defaultdict(lambda: [0, 0]), 0


def img(d, k):
    return {"type": "image", "image": Image.open(os.path.join(d, f"f{k}.jpg")).convert("RGB"), "max_pixels": MAXP, "min_pixels": MINP}


for i, q in enumerate(data, 1):
    cur = q["image"][-1]
    prior = list(range(max(1, cur - a.context), cur))
    content = []
    if prior:
        content.append({"type": "text", "text": "Earlier frames (oldest first):"})
        content += [img(q["video"], k) for k in prior]
        content.append({"type": "text", "text": "Current frame:"})
    for part in re.split(r"(<image>)", q["conversations"][0]["value"]):
        part = part.strip()
        if part == "<image>":
            content.append(img(q["video"], cur))
        elif part:
            content.append({"type": "text", "text": part})
    msgs = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFIX
    ims, _ = process_vision_info(msgs)
    inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
    resp = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    clean = resp.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    m = TAG.search(clean)
    pred = (LET.search(m.group(1)).group(1) if m and LET.search(m.group(1)) else "") or \
           (LET.search(re.sub(r"<\|[^>]*\|>", " ", clean)).group(1) if LET.search(re.sub(r"<\|[^>]*\|>", " ", clean)) else "")
    ok = pred == q["answer"]
    right += ok; by_verb[q["verb"]][0] += ok; by_verb[q["verb"]][1] += 1; n_think += "<think>" in (PREFIX + clean)
    fout.write(json.dumps({"id": q["id"], "answer": q["answer"], "pred": pred, "ok": ok, "verb": q["verb"],
                           "noun": q["noun"], "resp": clean[:400]}) + "\n")
    if i % 100 == 0 or i == len(data):
        print(f"{i}/{len(data)}  acc {100*right/i:.1f}%  thinking {n_think}/{i}", flush=True)
print(f"\nACC {100*right/len(data):.1f}% on {len(data)}  (chance 20%)")
for v, (c, t) in sorted(by_verb.items(), key=lambda x: -x[1][1])[:12]:
    print(f"  {v:22s} n={t:4d}  {100*c/t:5.1f}%")
print("example:", json.dumps(data[0]["options"]), "->", repr(clean[:200]))
