"""Rung 0 of the VLM ladder: zero-shot Qwen3-VL on the official Assembly101 anticipation set (camera C10119).
The model sees the 4 recent frames (the last 2 s before the cut; optionally the 8 history frames of the last 60 s
first) and the official vocabulary (17 verbs, 90 objects), and lists its 5 most likely next actions as
'verb | object'. Answers are mapped to the official classes (exact match, else closest name); each top-5 list is
filled up with the most frequent training classes if the model gave fewer than 5 valid ones. Metric: class-mean
top-5 recall for verb / object / action (same computation as the official code) + top-1 accuracy.
usage: python 02_zeroshot.py /workspace/data/a101_vlm /workspace/data/a101_anticip/data/CSVs OUT_DIR
                             [--history] [--limit 500] [--bs 8]
"""
import argparse, collections, csv, difflib, json, os, re, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("csvs"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--history", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ------------------------------------------------------------------ official class names (from the CSVs)
verb_name, noun_name, act_vn = {}, {}, {}
freq = [collections.Counter() for _ in range(3)]
for fn, ids, names, is_train in (("train.csv", ("action_id", "verb_id", "noun_id"), ("action_cls", "verb_cls", "noun_cls"), True),
                                 ("validation_challenge.csv", ("action", "verb", "noun"), ("action_cls", "verb_cls", "noun_cls"), False)):
    with open(os.path.join(a.csvs, fn)) as f:
        for r in csv.DictReader(f):
            ai, vi, ni = (int(r[k]) for k in ids)
            verb_name[vi] = r[names[1]].strip(); noun_name[ni] = r[names[2]].strip(); act_vn[(vi, ni)] = ai
            if is_train:
                for j, c in enumerate((ai, vi, ni)):
                    freq[j][c] += 1
verb_of = {v: k for k, v in verb_name.items()}; noun_of = {v: k for k, v in noun_name.items()}
prior = [[c for c, _ in f.most_common()] for f in freq]                 # fill-up order: most frequent first
VERBS = sorted(verb_name.values(), key=lambda v: -freq[1][verb_of[v]])
NOUNS = sorted(noun_name.values(), key=lambda n: -freq[2][noun_of[n]])
print(f"{len(VERBS)} verbs, {len(NOUNS)} objects, {len(act_vn)} verb-object actions", flush=True)

PROMPT = ("You are watching a person assemble or disassemble a toy vehicle from a fixed camera above the table. "
          "{what} Predict what the person will do next, starting about one second after the last frame.\n"
          "Allowed verbs: " + ", ".join(VERBS) + ".\nAllowed objects: " + ", ".join(NOUNS) + ".\n"
          "List the 5 most likely next actions, most likely first, one per line, exactly as: verb | object")


def closest(word, table):
    word = word.strip().lower()
    if word in table:
        return table[word]
    m = difflib.get_close_matches(word, list(table), n=1, cutoff=0.5)
    return table[m[0]] if m else None


def parse(text):
    vs, ns, acts = [], [], []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|-|\*)\s*", "", line)
        if "|" not in line:
            continue
        v, n = (p.strip() for p in line.split("|", 1))
        vi, ni = closest(v, verb_of), closest(n, noun_of)
        if vi is not None and vi not in vs:
            vs.append(vi)
        if ni is not None and ni not in ns:
            ns.append(ni)
        if vi is not None and ni is not None and (vi, ni) in act_vn and act_vn[(vi, ni)] not in acts:
            acts.append(act_vn[(vi, ni)])
    fill = lambda lst, j: (lst + [c for c in prior[j] if c not in lst])[:5]     # noqa: E731
    return fill(acts, 0), fill(vs, 1), fill(ns, 2), (len(acts), len(vs), len(ns))


def cm_recall(top5s, ys):
    per = collections.defaultdict(list)
    for t, y in zip(top5s, ys):
        per[y].append(y in t)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
data = json.load(open(os.path.join(a.vlm, "validation.json")))[: a.limit or None]
img = lambda q, f: Image.open(os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")).convert("RGB")   # noqa: E731

done = {}
pf = os.path.join(a.out, "answers.jsonl")
if os.path.exists(pf):
    for line in open(pf):
        r = json.loads(line); done[r["id"]] = r
fout = open(pf, "a")
todo = [q for q in data if q["id"] not in done]
print(f"{len(data)} validation segments, {len(todo)} to do | history frames: {a.history}", flush=True)
t0 = time.time()
for s in range(0, len(todo), a.bs):
    batch = todo[s:s + a.bs]
    msgs = []
    for q in batch:
        content = []
        if a.history and q["hist"]:
            content.append({"type": "text", "text": f"Earlier frames from the last minute ({len(q['hist'])}, oldest first):"})
            content += [{"type": "image", "image": img(q, f)} for f in q["hist"]]
        content.append({"type": "text", "text": "The last 2 seconds (4 frames, oldest first):"})
        content += [{"type": "image", "image": img(q, f)} for f in q["recent"]]
        what = "You see the last minute and the last 2 seconds." if a.history else "You see the last 2 seconds."
        content.append({"type": "text", "text": PROMPT.format(what=what)})
        msgs.append([{"role": "user", "content": content}])
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
    texts = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    for q, t in zip(batch, texts):
        r = {"id": q["id"], "y": q["y"], "text": t}
        done[q["id"]] = r; fout.write(json.dumps(r) + "\n")
    fout.flush()
    n = s + len(batch)
    if (s // a.bs) % 10 == 0 or n == len(todo):
        el = time.time() - t0
        print(f"{n}/{len(todo)} | {el / n:.2f} s/segment, ~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)

recs = [done[q["id"]] for q in data]
parsed = [parse(r["text"]) for r in recs]
res = {}
for j, k in enumerate(("action", "verb", "noun")):
    tops = [p[j] for p in parsed]; ys = [r["y"][j] for r in recs]
    res[k] = {"cm_top5_recall": cm_recall(tops, ys), "top1": 100 * sum(t[0] == y for t, y in zip(tops, ys)) / len(ys)}
valid = [sum(p[3][j] > 0 for p in parsed) / len(parsed) for j in range(3)]
json.dump({"args": vars(a), "results": res, "share_with_valid_answer": valid}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(f"\nzero-shot {a.model} | history {a.history} | {len(recs)} segments")
for k in ("verb", "noun", "action"):
    print(f"  {k:6s} class-mean top-5 recall {res[k]['cm_top5_recall']:5.1f} | top-1 {res[k]['top1']:5.1f}")
print(f"  answers with at least one valid action / verb / object: {valid[0]:.0%} / {valid[1]:.0%} / {valid[2]:.0%}")
print("  example answer:", repr(recs[0]["text"][:200]))
