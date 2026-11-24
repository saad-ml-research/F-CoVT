"""Language reasoning for anticipation with Qwen3-VL (camera C10119). Inputs per segment: the 4 recent frames (last 2 s)
and, if used, the PREDICTED past actions as text (06_recog.py top-1, last 8 steps; TRUE ones in hindsight mode).
modes
  zs         idea 1: zero-shot on --n validation segments, prompt variants (--variants):
               plain   frames only, list 5 "verb | object"
               hist    + predicted history text
               reason  + history, first 1-2 sentences "where are we, what is next", then the list
               rnohist reasoning without history
             -> answers + class-mean recall / top-1 / top-5 acc per variant (same parsing as 02_zeroshot.py)
  thoughts   idea 2: the 'reason' prompt on a split; keeps the generated text and the layer-L hidden states over the
             generated tokens: [mean over the rationale+answer tokens, final token] -> chunks <out>/<split>_<k>.pt
  hindsight  idea 3a: frames + TRUE history + the TRUE next action; the model explains in 1-2 sentences why it is the
             natural next step -> <out>/hindsight_<split>.jsonl (training targets for a student without the future)
--subset N picks a fixed random subset of the split (seed 0; same ids in every mode) and saves it.
usage: python 13_reason.py MODE /workspace/data/a101_vlm /workspace/data/a101_anticip/data/CSVs
                           /workspace/outputs/recog256v/recog.pt OUT [--split train] [--subset 12000] [--n 1000]
"""
import argparse, collections, csv, difflib, json, math, os, re, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["zs", "thoughts", "hindsight"])
ap.add_argument("vlm"); ap.add_argument("csvs"); ap.add_argument("recog"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--split", default="validation")
ap.add_argument("--subset", type=int, default=0)
ap.add_argument("--n", type=int, default=1000, help="zs: number of validation segments")
ap.add_argument("--variants", default="plain,hist,reason,rnohist")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ------------------------------------------------------------------ names, priors, mapping (as 02_zeroshot.py)
verb_name, noun_name, act_vn, VN = {}, {}, {}, {}
freq = [collections.Counter() for _ in range(3)]
for fn, ids, is_train in (("train.csv", ("action_id", "verb_id", "noun_id"), True), ("validation_challenge.csv", ("action", "verb", "noun"), False)):
    with open(os.path.join(a.csvs, fn)) as f:
        for r in csv.DictReader(f):
            ai, vi, ni = (int(r[k]) for k in ids)
            v, o = r["verb_cls"].strip(), r["noun_cls"].strip()
            verb_name[vi] = v; noun_name[ni] = o; act_vn[(vi, ni)] = ai; VN[ai] = f"{v} {o}".replace("_", " ")
            if is_train and r["video"].endswith("/C10119_rgb.mp4"):
                for j, c in enumerate((ai, vi, ni)):
                    freq[j][c] += 1
verb_of = {v: k for k, v in verb_name.items()}; noun_of = {v: k for k, v in noun_name.items()}
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


# ------------------------------------------------------------------ data
R = torch.load(a.recog)
IDX = {s: json.load(open(os.path.join(a.vlm, f"{s}.json"))) for s in ("train", "validation")}
for s in IDX:
    assert [q["y"] for q in IDX[s]] == R[s]["y"].tolist(), f"{s}: frame index and recognizer differ"


def pick(split, n):
    g = torch.Generator().manual_seed(0)
    return sorted(torch.randperm(len(IDX[split]), generator=g)[:n].tolist()) if n else list(range(len(IDX[split])))


def history_text(split, i, true=False):
    H = R[split]["hist"][i, -a.K:].tolist()
    lab = R[split]["y"][:, 0] if true else R[split]["pred"][:, 0]
    steps = [VN.get(int(lab[j]), "unknown action") for j in H if j >= 0]
    return "; ".join(steps) if steps else "nothing yet"


FMT = ("Allowed verbs: " + ", ".join(VERBS) + ".\nAllowed objects: " + ", ".join(NOUNS) + ".\n"
       "List the 5 most likely next actions, most likely first, one per line, exactly as: verb | object")
INTRO = "You are watching a person assemble or disassemble a toy vehicle from a fixed camera above the table. "


def prompt(variant, split, i):
    q = IDX[split][i]
    content = [{"type": "text", "text": "The last 2 seconds (4 frames, oldest first):"}]
    content += [{"type": "image", "image": Image.open(os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")).convert("RGB")}
                for f in q["recent"]]
    if variant in ("hist", "reason"):
        hist = f"Earlier steps, oldest first (recognized automatically, may contain mistakes): {history_text(split, i)}.\n"
    elif variant == "hindsight":
        hist = f"Earlier steps, oldest first: {history_text(split, i, true=True)}.\n"
    else:
        hist = ""
    if variant in ("plain", "hist"):
        task = INTRO + hist + "Predict what the person will do next, starting about one second after the last frame.\n" + FMT
    elif variant in ("reason", "rnohist"):
        task = (INTRO + hist + "First, in one or two short sentences, say where the person is in the assembly and what the "
                "next step is likely to be. Then write 'Answer:' on its own line, followed by the list.\n" + FMT)
    else:   # hindsight
        task = (INTRO + hist + f"The person's next action (about one second after the last frame) is: {VN[q['y'][0]]}.\n"
                "In one or two short sentences, explain why this is the natural next step, using the earlier steps and "
                "what you see. Do not mention that you were told the answer.")
    content.append({"type": "text", "text": task})
    return [{"role": "user", "content": content}]


proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
EOS = {proc.tokenizer.eos_token_id, proc.tokenizer.convert_tokens_to_ids("<|im_end|>")}


@torch.no_grad()
def generate(msgs, max_new, keep_states=False):
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, output_hidden_states=keep_states,
                         return_dict_in_generate=True)
    seq = out.sequences[:, inp["input_ids"].shape[1]:]
    texts = proc.batch_decode(seq, skip_special_tokens=True)
    if not keep_states:
        return texts, None
    # hidden states of the generated tokens: steps 1.. hold [B, 1, D] per layer (step 0 is the prompt)
    H = torch.stack([st[a.layer][:, -1] for st in out.hidden_states[1:]], 1).float()      # [B, T-1, D]
    valid = torch.ones(seq.shape, dtype=torch.bool, device=seq.device)
    for b in range(len(seq)):
        ends = [t for t, x in enumerate(seq[b].tolist()) if x in EOS]
        if ends:
            valid[b, ends[0]:] = False
    valid = valid[:, :H.shape[1]]
    n = valid.sum(1).clamp(min=1)
    mean = (H * valid[..., None]).sum(1) / n[:, None]
    last = H[torch.arange(len(H)), (n - 1).clamp(max=H.shape[1] - 1)]
    return texts, torch.stack([mean, last], 1).half().cpu()


t0 = time.time()
if a.mode == "zs":
    ids = pick("validation", a.n)
    ys = [IDX["validation"][i]["y"] for i in ids]
    summary = {}
    for variant in a.variants.split(","):
        path = os.path.join(a.out, f"zs_{variant}.jsonl")
        done = {json.loads(l)["i"]: json.loads(l)["text"] for l in open(path)} if os.path.exists(path) else {}
        f = open(path, "a")
        todo = [i for i in ids if i not in done]
        for s in range(0, len(todo), a.bs):
            b = todo[s:s + a.bs]
            texts, _ = generate([prompt(variant, "validation", i) for i in b], 200 if "reason" in variant or variant == "rnohist" else 80)
            for i, t in zip(b, texts):
                done[i] = t; f.write(json.dumps({"i": i, "text": t}) + "\n")
            f.flush()
        tops = [parse(done[i].split("Answer:")[-1]) for i in ids]
        summary[variant] = metrics(tops, ys)
        m = summary[variant]
        print(f"{variant:8s} ({len(ids)} val segments): recall verb {m['verb']:5.1f} object {m['noun']:5.1f} action {m['action']:5.1f}"
              f" | action top-1 {m['action_top1']:4.1f} top-5 acc {m['action_top5']:4.1f} | {(time.time() - t0) / 60:.0f} min", flush=True)
        print(f"   example: {done[ids[3]][:300]!r}", flush=True)
    json.dump(summary, open(os.path.join(a.out, "zs_summary.json"), "w"), indent=1)

elif a.mode == "hindsight":
    ids = pick(a.split, a.subset)
    json.dump(ids, open(os.path.join(a.out, f"subset_{a.split}.json"), "w"))
    path = os.path.join(a.out, f"hindsight_{a.split}.jsonl")
    done = {json.loads(l)["i"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    todo = [i for i in ids if i not in done]
    for s in range(0, len(todo), a.bs):
        b = todo[s:s + a.bs]
        texts, _ = generate([prompt("hindsight", a.split, i) for i in b], 80)
        for i, t in zip(b, texts):
            f.write(json.dumps({"i": i, "y": IDX[a.split][i]["y"], "rationale": t.strip()}) + "\n")
        f.flush()
        if (s // a.bs) % 25 == 0:
            el = time.time() - t0; k = s + len(b)
            print(f"hindsight {a.split}: {k}/{len(todo)} | ~{el / k * (len(todo) - k) / 60:.0f} min left | e.g. {texts[0][:160]!r}", flush=True)

else:   # thoughts
    ids = pick(a.split, a.subset)
    json.dump(ids, open(os.path.join(a.out, f"subset_{a.split}.json"), "w"))
    for k in range(math.ceil(len(ids) / a.chunk)):
        path = os.path.join(a.out, f"{a.split}_{k}.pt")
        if os.path.exists(path):
            continue
        part = ids[k * a.chunk:(k + 1) * a.chunk]
        F_, T_ = [], []
        for s in range(0, len(part), a.bs):
            b = part[s:s + a.bs]
            texts, st = generate([prompt("reason", a.split, i) for i in b], 200, keep_states=True)
            F_.append(st); T_ += texts
            if (s // a.bs) % 25 == 0:
                el = time.time() - t0
                print(f"thoughts {a.split} chunk {k + 1}: {s + len(b)}/{len(part)} | {el / 60:.0f} min elapsed", flush=True)
        ys = [IDX[a.split][i]["y"] for i in part]
        m = metrics([parse(t.split("Answer:")[-1]) for t in T_], ys)
        torch.save({"ids": part, "feats": torch.cat(F_), "texts": T_, "y": torch.tensor(ys), "layer": a.layer}, path)
        print(f"saved {path} | its answers: recall action {m['action']:.1f} verb {m['verb']:.1f} object {m['noun']:.1f}", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min")
