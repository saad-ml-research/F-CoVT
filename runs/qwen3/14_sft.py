"""Idea 3b, hindsight rationale distillation: LoRA-tune Qwen3-VL to REASON about the next step without seeing the future.
Teacher side (13_reason.py hindsight): frames + true history + the TRUE next action -> a 1-2 sentence rationale.
Student (this script): frames + PREDICTED history (the fair 'reason' prompt of 13_reason.py) -> target text =
  <hindsight rationale>\nAnswer:\n<true next action>\n<4 plausible others: the actions that most often follow the
  last true past action in train>
Trained on the hindsight subset of train (1 epoch, LoRA rank 16 on q/k/v/o); then generates on the same 1,000
validation segments as 13_reason.py zs, so it compares directly with zero-shot 'reason' (4.5) and 'hist' (5.6).
usage: python 14_sft.py /workspace/data/a101_vlm /workspace/data/a101_anticip/data/CSVs /workspace/outputs/recog256v/recog.pt
                        /workspace/outputs/q3_reason OUT [--max_train 0] [--n_eval 1000]
"""
import argparse, collections, csv, difflib, json, math, os, re, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("csvs"); ap.add_argument("recog"); ap.add_argument("reason_dir"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--n_eval", type=int, default=1000)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--no_rationale", action="store_true", help="control: target = the answer list only, prompt asks only for the list")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.txt"), "a")


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()


# ------------------------------------------------------------------ names, priors, parsing (as 13_reason.py)
verb_name, noun_name, act_vn, VN, VO = {}, {}, {}, {}, {}
freq = [collections.Counter() for _ in range(3)]
for fn, ids, is_train in (("train.csv", ("action_id", "verb_id", "noun_id"), True), ("validation_challenge.csv", ("action", "verb", "noun"), False)):
    with open(os.path.join(a.csvs, fn)) as f:
        for r in csv.DictReader(f):
            ai, vi, ni = (int(r[k]) for k in ids)
            v, o = r["verb_cls"].strip(), r["noun_cls"].strip()
            verb_name[vi] = v; noun_name[ni] = o; act_vn[(vi, ni)] = ai; VN[ai] = f"{v} {o}".replace("_", " ")
            VO[ai] = (vi, ni)
            if is_train and r["video"].endswith("/C10119_rgb.mp4"):
                for j, c in enumerate((ai, vi, ni)):
                    freq[j][c] += 1
verb_of = {v: k for k, v in verb_name.items()}; noun_of = {v: k for k, v in noun_name.items()}
vn_of = VO                     # action id -> (verb id, object id), from the annotation rows (several ids can share a pair)
prior = [[c for c, _ in f.most_common()] for f in freq]
VERBS = sorted(verb_name.values(), key=lambda v: -freq[1][verb_of[v]])
NOUNS = sorted(noun_name.values(), key=lambda n: -freq[2][noun_of[n]])


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
    return fill(acts, 0), fill(vs, 1), fill(ns, 2)


def metrics(tops, ys):
    out = {}
    for j, k in enumerate(("action", "verb", "noun")):
        per = collections.defaultdict(list)
        for t, y in zip(tops, ys):
            per[y[j]].append(y[j] in t[j])
        out[k] = 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)
        out[k + "_top1"] = 100 * sum(t[j][0] == y[j] for t, y in zip(tops, ys)) / len(ys)
        out[k + "_top5"] = 100 * sum(y[j] in t[j] for t, y in zip(tops, ys)) / len(ys)
    return out


R = torch.load(a.recog)
IDX = {s: json.load(open(os.path.join(a.vlm, f"{s}.json"))) for s in ("train", "validation")}

# transition statistics on TRUE train labels: last past action -> most common next actions (for the 4 alternatives)
nxt = collections.defaultdict(collections.Counter)
Htr = R["train"]["hist"][:, -1]
for i in range(len(Htr)):
    if Htr[i] >= 0:
        nxt[int(R["train"]["y"][Htr[i], 0])][int(R["train"]["y"][i, 0])] += 1


def history_text(split, i):
    H = R[split]["hist"][i, -a.K:].tolist()
    steps = [VN.get(int(R[split]["pred"][j, 0]), "unknown action") for j in H if j >= 0]
    return "; ".join(steps) if steps else "nothing yet"


FMT = ("Allowed verbs: " + ", ".join(VERBS) + ".\nAllowed objects: " + ", ".join(NOUNS) + ".\n"
       "List the 5 most likely next actions, most likely first, one per line, exactly as: verb | object")


def user_msg(split, i):
    q = IDX[split][i]
    content = [{"type": "text", "text": "The last 2 seconds (4 frames, oldest first):"}]
    content += [{"type": "image", "image": Image.open(os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")).convert("RGB")}
                for f in q["recent"]]
    content.append({"type": "text", "text":
        "You are watching a person assemble or disassemble a toy vehicle from a fixed camera above the table. "
        f"Earlier steps, oldest first (recognized automatically, may contain mistakes): {history_text(split, i)}.\n"
        + ("Predict what the person will do next, starting about one second after the last frame. Write 'Answer:' on its own "
           "line, followed by the list.\n" if a.no_rationale else
           "First, in one or two short sentences, say where the person is in the assembly and what the next step is likely "
           "to be. Then write 'Answer:' on its own line, followed by the list.\n") + FMT})
    return {"role": "user", "content": content}


def target(i, rationale):
    y = int(R["train"]["y"][i, 0])
    H = int(R["train"]["hist"][i, -1])
    alts = [c for c, _ in nxt[int(R["train"]["y"][H, 0])].most_common()] if H >= 0 else []
    alts = [c for c in alts + prior[0] if c != y and c in vn_of]
    lines = [y] + list(dict.fromkeys(alts))[:4]
    return ("" if a.no_rationale else rationale.strip() + "\n") + "Answer:\n" + "\n".join(f"{verb_name[vn_of[c][0]]} | {noun_name[vn_of[c][1]]}" for c in lines)


data = [json.loads(l) for l in open(os.path.join(a.reason_dir, "hindsight_train.jsonl"))]
data = [d for d in data if d["rationale"]][: a.max_train or None]
say(f"{len(data)} hindsight examples | e.g. target:\n{target(data[0]['i'], data[0]['rationale'])}")

proc = AutoProcessor.from_pretrained(a.model)
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda")
for p in model.parameters():
    p.requires_grad_(False)
peft = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
params = [p for p in model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
say(f"LoRA params {sum(p.numel() for p in params) / 1e6:.1f} M")
ASSIST = proc.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(data) / (a.bs * a.accum))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

# ------------------------------------------------------------------ train (1 epoch)
proc.tokenizer.padding_side = "right"
model.train()
g = torch.Generator().manual_seed(0)
order = torch.randperm(len(data), generator=g).tolist()
t0, step = time.time(), 0
for k in range(0, len(order), a.bs):
    batch = [data[j] for j in order[k:k + a.bs]]
    convs = [[user_msg("train", d["i"]), {"role": "assistant", "content": [{"type": "text", "text": target(d["i"], d["rationale"])}]}]
             for d in batch]
    inp = proc.apply_chat_template(convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to("cuda")
    labels = inp["input_ids"].clone()
    for b in range(len(batch)):
        ids = labels[b].tolist()
        st = max(i for i in range(len(ids) - len(ASSIST) + 1) if ids[i:i + len(ASSIST)] == ASSIST) + len(ASSIST)
        labels[b, :st] = -100
    labels[inp["attention_mask"] == 0] = -100
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inp, labels=labels).loss
    (loss / a.accum).backward()
    if (k // a.bs + 1) % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            say(f"step {step}/{steps} | loss {loss.item():.3f} | {(k + len(batch)) / el:.2f} examples/s | ~{(steps - step) * el / step / 60:.0f} min left")
peft.save_pretrained(os.path.join(a.out, "lora"))

# ------------------------------------------------------------------ evaluate: generate on the same 1,000 validation segments
model.eval(); proc.tokenizer.padding_side = "left"
gv = torch.Generator().manual_seed(0)
ids = sorted(torch.randperm(len(IDX["validation"]), generator=gv)[: a.n_eval].tolist())
texts = []
with torch.no_grad():
    for k in range(0, len(ids), 16):
        b = ids[k:k + 16]
        inp = proc.apply_chat_template([[user_msg("validation", i)] for i in b], tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        texts += proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
json.dump([{"i": i, "text": t} for i, t in zip(ids, texts)], open(os.path.join(a.out, "eval_texts.json"), "w"), indent=1)
m = metrics([parse(t.split("Answer:")[-1]) for t in texts], [IDX["validation"][i]["y"] for i in ids])
json.dump(m, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
say(f"student after hindsight SFT ({len(ids)} val segments): recall verb {m['verb']:.1f} object {m['noun']:.1f} action {m['action']:.1f}"
    f" | action top-1 {m['action_top1']:.1f} top-5 acc {m['action_top5']:.1f}   (zero-shot: reason 4.5, hist 5.6)"
    f"{'   [CONTROL: no rationale]' if a.no_rationale else ''}")
say(f"example: {texts[3][:400]!r}")
say(f"done in {(time.time() - t0) / 60:.0f} min")
