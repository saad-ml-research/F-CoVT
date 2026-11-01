"""Evaluate a run trained with FCOVT_HEAD=1 on the Assembly101 classification-head questions.
Prompt = training layout (see 03_eval_mcq.py); the reply is then built and the head reads the hidden state of its
last token, the end of "<answer>":
  --think none      reply = "<answer>"                                   (rows B, F direct)
  --think latent    reply = forced latent blocks + "<answer>"            (row D; --blocks 2)
  --think generate  the model writes its own thoughts until "<answer>"   (row F thinking)
Metrics: top-1 / top-5 accuracy and the official class-mean top-5 recall, for action, verb and object.
usage: python 06_eval_head.py RUN_DIR /workspace/data/a101/validation_head.json OUT_DIR --think none [--limit 500]
       (RUN_DIR holds merged/ and act_head.pt)
"""
import argparse, collections, json, os, re, sys
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fcovt"))
import fcovt_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--think", default="none", choices=["none", "latent", "generate"])
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--context", type=int, default=4)
ap.add_argument("--blind", action="store_true")
a = ap.parse_args()
MAXP, MINP = 256 * 28 * 28, 64 * 28 * 28

proc = AutoProcessor.from_pretrained(os.path.join(a.run, "merged"))
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(a.run, "merged"), torch_dtype=torch.bfloat16,
                                                           attn_implementation="flash_attention_2", device_map="cuda").eval()
H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
head = fcovt_core.ActHead(H)
head.load_state_dict(torch.load(os.path.join(a.run, "act_head.pt"), map_location="cpu"))
head = head.to("cuda", torch.bfloat16).eval()
data = json.load(open(a.data))[: a.limit or None]
os.makedirs(a.out, exist_ok=True)


def img(d, k):
    im = Image.open(os.path.join(d, f"f{k}.jpg")).convert("RGB")
    if a.blind:
        im = Image.new("RGB", im.size)
    return {"type": "image", "image": im, "max_pixels": MAXP, "min_pixels": MINP}


def prompt(q):
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
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), process_vision_info(msgs)[0]


latent = "".join(fcovt_core.block("generic") for _ in range(a.blocks))
res = collections.defaultdict(list)          # kind -> [(label, top5 list)]
fout = open(os.path.join(a.out, "preds.jsonl"), "w")
n_think = 0
for i, q in enumerate(data, 1):
    text, ims = prompt(q)
    if a.think == "latent":
        text += latent + "<answer>"
    elif a.think == "none":
        text += "<answer>"
    else:
        inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=320, do_sample=False, stop_strings=["<answer>"], tokenizer=proc.tokenizer)
        gen = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        gen = gen.split("<answer>")[0].replace("<|im_end|>", "")
        n_think += "<think>" in gen
        text += gen + "<answer>"
    inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
    with torch.no_grad():
        h = model(**inp, output_hidden_states=True).hidden_states[-1][:, -1]
        la, lv, ln = head(h.to(torch.bfloat16))
    rec = {"id": q["id"]}
    for kind, logits, y in (("action", la, q["action_id"]), ("verb", lv, q["verb_id"]), ("noun", ln, q["noun_id"])):
        top5 = logits[0].float().topk(5).indices.tolist()
        res[kind].append((y, top5)); rec[kind] = {"y": y, "top5": top5}
    fout.write(json.dumps(rec) + "\n")
    if i % 100 == 0 or i == len(data):
        r = res["action"]
        print(f"{i}/{len(data)}  action top1 {100*sum(t[0]==y for y,t in r)/len(r):.1f}%  top5 {100*sum(y in t for y,t in r)/len(r):.1f}%"
              + (f"  thinking {n_think}/{i}" if a.think == "generate" else ""), flush=True)

summary = {}
print(f"\n{a.run}  think={a.think}  n={len(data)}")
for kind, r in res.items():
    top1 = 100 * sum(t[0] == y for y, t in r) / len(r)
    top5 = 100 * sum(y in t for y, t in r) / len(r)
    per = collections.defaultdict(list)
    for y, t in r:
        per[y].append(y in t)
    mcr = 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)
    summary[kind] = {"top1": top1, "top5": top5, "class_mean_top5_recall": mcr, "classes": len(per)}
    print(f"  {kind:6s} top1 {top1:5.1f}  top5 {top5:5.1f}  class-mean top5 recall {mcr:5.1f}  ({len(per)} classes)")
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
