"""Rebuild the 5 options of existing Assembly101 questions (same ids, same frames) so the text alone gives nothing away.
v1 distractors shared the answer's verb or object, which made the answer the option overlapping most with the others
(blind B ~80%). v2: distractors are PLAUSIBLE NEXT ACTIONS, i.e. actions that really follow the previous action
somewhere in the TRAIN recordings, sampled by how often they follow it, chosen without looking at the answer's words;
filled up with frequent actions of the same toy. Also reports two text-only cheats per version (should be ~20% for v2):
  overlap  = pick the option sharing the most verbs/objects with the other options
  frequent = pick the option that is the most frequent action in train
usage: python 04_reoption.py /workspace/data/assembly101/annotations /workspace/data/a101   (writes <split>_v2.json)
"""
import collections, csv, json, os, random, sys

ann, d = sys.argv[1], sys.argv[2]
VIEW = "C10119_rgb.mp4"
L = "ABCDE"
PROMPT = ("Select the best answer to the following multiple-choice question based on the frames. "
          "What will the person do next?\n{opts}\nPlease provide only the single option letter within the <answer> </answer> tags.")


def rows(split):
    out = collections.defaultdict(list)
    with open(os.path.join(ann, "fine-grained-annotations", f"{split}.csv")) as f:
        for r in csv.DictReader(f):
            if r["video"].endswith("/" + VIEW) and r.get("action_cls"):
                out[r["video"].split("/")[0]].append((int(r["start_frame"]), int(r["end_frame"]), r["action_cls"],
                                                      r["verb_cls"], r["noun_cls"], r["toy_id"]))
    for v in out.values():
        v.sort()
    return out


train_rows = rows("train")
follows = collections.defaultdict(collections.Counter)      # prev action -> Counter(next action), train only
freq, toy_freq, vn = collections.Counter(), collections.defaultdict(collections.Counter), {}
for segs in train_rows.values():
    for a, b in zip(segs, segs[1:]):
        follows[a[2]][b[2]] += 1
    for s in segs:
        freq[s[2]] += 1; toy_freq[s[5]][s[2]] += 1; vn[s[2]] = (s[3], s[4])


def overlap_pick(opts):
    sc = [sum((vn.get(o, ("", ""))[0] == vn.get(p, ("", ""))[0]) + (vn.get(o, ("", ""))[1] == vn.get(p, ("", ""))[1])
              for p in opts if p != o) for o in opts]
    return opts[max(range(len(opts)), key=lambda i: sc[i])]


def frequent_pick(opts):
    return max(opts, key=lambda o: freq[o])


def wsample(counter, k, exclude, rng):
    items = [(a, c) for a, c in counter.items() if a not in exclude]
    out = []
    while items and len(out) < k:
        tot = sum(c for _, c in items); x = rng.random() * tot
        for i, (a, c) in enumerate(items):
            x -= c
            if x <= 0:
                out.append(a); items.pop(i); break
    return out


rng = random.Random(0)
for split in ("train", "validation", "test"):
    p = os.path.join(d, f"{split}.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(ann, "fine-grained-annotations", f"{split}.csv")):
        continue
    qs, segs_by = json.load(open(p)), rows(split)
    cheat = collections.Counter(); src = collections.Counter()
    for q in qs:
        ans = q["answer_text"]
        cheat["v1_overlap"] += overlap_pick(list(q["options"].values())) == ans
        cheat["v1_frequent"] += frequent_pick(list(q["options"].values())) == ans
        segs = segs_by[q["seq"]]
        prev = [s for s in segs if s[0] < q["next_start"]]
        prev = prev[-1][2] if prev else None
        dis = wsample(follows.get(prev, {}), 4, {ans}, rng); src["follows"] += len(dis)
        if len(dis) < 4:
            more = wsample(toy_freq[q["toy_id"]], 4 - len(dis), {ans, *dis}, rng); src["toy_freq"] += len(more); dis += more
        if len(dis) < 4:
            more = wsample(freq, 4 - len(dis), {ans, *dis}, rng); src["global"] += len(more); dis += more
        opts = dis + [ans]; rng.shuffle(opts)
        q["options"] = {L[i]: o for i, o in enumerate(opts)}
        q["answer"] = L[opts.index(ans)]
        q["prev_action"] = prev
        q["conversations"][0]["value"] = "frame_1\n<image>\n" + PROMPT.format(
            opts="\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)))
        cheat["v2_overlap"] += overlap_pick(opts) == ans
        cheat["v2_frequent"] += frequent_pick(opts) == ans
    json.dump(qs, open(os.path.join(d, f"{split}_v2.json"), "w"))
    n = len(qs)
    print(f"{split}: {n} questions | text-only cheats: " + ", ".join(f"{k} {100*v/n:.1f}%" for k, v in sorted(cheat.items())) +
          f" | distractor sources {dict(src)} | letters {dict(sorted(collections.Counter(q['answer'] for q in qs).items()))}")
    print("  example:", qs[0]["prev_action"], "->", json.dumps(qs[0]["options"]), "answer", qs[0]["answer"])
