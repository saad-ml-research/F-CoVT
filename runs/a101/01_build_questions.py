"""Assembly101 -> "what will the person do next?" multiple-choice questions (annotations only, no video).

For every fine-grained action j in camera C10119, cut the video TAU before j starts:
    input  = 5 stills at cut-4s, -3s, -2s, -1s, cut          -> f1..f5 (f5 = current frame)
    future = a frame inside action j (middle, at most 1 s in) -> f6   (near future; graded, never input)
             the end of the coarse step that action j belongs to -> f7   (step result; graded, never input)
    answer = j's action name, as 1 of 5 options (A-E)
Skipped: another action starts inside (cut, start_j] (ambiguous "next"); j repeats the action running at the cut;
less than 4 s of video before the cut; no coarse step covers j.
Distractors: actions seen with the same toy, preferring the same verb or object (hard), else any action of that toy.

Output: <out>/<split>.json in the Future-L1 loader format (video = frames folder, image=[5], reasoning_image=[6, 7],
answer = the option letter; action_id / verb_id / noun_id for a classification head), plus <out>/<split>_frames.json = which 30 fps frame numbers to extract per question.
usage: python 01_build_questions.py /workspace/data/assembly101/annotations /workspace/data/a101 --train_max 10000 --val_max 2000 [--test_max 2000]
"""
import argparse, csv, json, os, random, collections

ap = argparse.ArgumentParser()
ap.add_argument("ann"); ap.add_argument("out")
ap.add_argument("--view", default="C10119_rgb.mp4")
ap.add_argument("--tau", type=float, default=1.0, help="seconds between the cut and the next action")
ap.add_argument("--train_max", type=int, default=10000); ap.add_argument("--val_max", type=int, default=2000)
ap.add_argument("--test_max", type=int, default=0, help="0 = skip test (official test labels may be hidden)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frames_root", default="/workspace/data/a101/frames")
a = ap.parse_args()
FPS = 30; TAU = int(round(a.tau * FPS)); OBS = [4 * FPS, 3 * FPS, 2 * FPS, 1 * FPS, 0]
L = "ABCDE"
PROMPT = ("Select the best answer to the following multiple-choice question based on the frames. "
          "What will the person do next?\n{opts}\nPlease provide only the single option letter within the <answer> </answer> tags.")


def load_coarse():
    """seq -> sorted [(start, end, name)] from coarse_labels/{assembly,disassembly}_<seq>.txt (30 fps frames)."""
    d = os.path.join(a.ann, "coarse-annotations", "coarse_labels")
    out = collections.defaultdict(list)
    for fn in os.listdir(d):
        seq = fn.split("_", 1)[1].rsplit(".", 1)[0]
        for line in open(os.path.join(d, fn)):
            p = line.split()
            if len(p) >= 3:
                out[seq].append((int(p[0]), int(p[1]), " ".join(p[2:])))
    for v in out.values():
        v.sort()
    return out


COARSE = load_coarse()


def load(split):
    rows = []
    if not os.path.exists(os.path.join(a.ann, "fine-grained-annotations", f"{split}.csv")):
        return rows
    with open(os.path.join(a.ann, "fine-grained-annotations", f"{split}.csv")) as f:
        for r in csv.DictReader(f):
            if r["video"].endswith("/" + a.view) and r.get("action_cls"):  # rows without labels are skipped
                rows.append(dict(seq=r["video"].split("/")[0], s=int(r["start_frame"]), e=int(r["end_frame"]),
                                 act=r["action_cls"], verb=r["verb_cls"], noun=r["noun_cls"], toy=r["toy_id"],
                                 action_id=int(r["action_id"] or -1), verb_id=int(r["verb_id"] or -1),
                                 noun_id=int(r["noun_id"] or -1)))
    return rows


def build(split, cap, rng):
    rows = load(split)
    if not rows:
        print(f"{split}: no rows for {a.view}"); return []
    by_seq = collections.defaultdict(list)
    toy_acts = collections.defaultdict(set)
    for r in rows:
        by_seq[r["seq"]].append(r); toy_acts[r["toy"]].add((r["act"], r["verb"], r["noun"]))
    all_acts = sorted({(r["act"], r["verb"], r["noun"]) for r in rows})
    qs, skip = [], collections.Counter()
    for seq, segs in by_seq.items():
        segs.sort(key=lambda r: (r["s"], r["e"]))
        starts = [r["s"] for r in segs]
        for j, nxt in enumerate(segs):
            cut = nxt["s"] - TAU
            if cut - OBS[0] < 0: skip["too_early"] += 1; continue
            if any(cut < st < nxt["s"] for st in starts) or starts.count(nxt["s"]) > 1:  # another action starts in the gap or at the same frame
                skip["ambiguous_next"] += 1; continue
            running = [r for r in segs if r["s"] <= cut < r["e"]]
            if any(r["act"] == nxt["act"] for r in running): skip["repeat"] += 1; continue
            step = next((c for c in COARSE.get(seq, []) if c[0] <= nxt["s"] < c[1]), None)
            if step is None: skip["no_coarse"] += 1; continue
            pool = [x for x in toy_acts[nxt["toy"]] if x[0] != nxt["act"]]
            hard = [x for x in pool if x[1] == nxt["verb"] or x[2] == nxt["noun"]]
            rng.shuffle(hard); rng.shuffle(pool)
            dis = []
            for x in hard + pool + rng.sample(all_acts, min(50, len(all_acts))):
                if x[0] != nxt["act"] and x[0] not in dis: dis.append(x[0])
                if len(dis) == 4: break
            if len(dis) < 4: skip["few_distractors"] += 1; continue
            opts = dis + [nxt["act"]]; rng.shuffle(opts)
            ans = L[opts.index(nxt["act"])]
            fut = nxt["s"] + min(FPS, max(1, (nxt["e"] - nxt["s"]) // 2))
            res = max(fut + FPS // 2, step[1] - 3)          # end of the step, at least 0.5 s after the near frame
            qid = f"{seq}__{nxt['s']:07d}"
            qs.append({
                "id": qid, "seq": seq, "split": split, "toy_id": nxt["toy"],
                "video": os.path.join(a.frames_root, qid),
                "image": [5], "reasoning_image": [6, 7],
                "frames_30fps": [cut - o for o in OBS] + [fut, res],
                "conversations": [{"from": "human", "value": "frame_1\n<image>\n" + PROMPT.format(
                    opts="\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)))},
                                  {"from": "gpt", "value": ""}],
                "options": {L[i]: o for i, o in enumerate(opts)}, "answer": ans, "answer_text": nxt["act"],
                "action_id": nxt["action_id"], "verb_id": nxt["verb_id"], "noun_id": nxt["noun_id"],
                "verb": nxt["verb"], "noun": nxt["noun"], "step": step[2], "step_end": step[1],
                "running": [r["act"] for r in running], "cut_frame": cut, "next_start": nxt["s"]})
    assert len({q["id"] for q in qs}) == len(qs), "duplicate question ids"
    rng.shuffle(qs)
    if qs:
        gap = sorted((q["frames_30fps"][6] - q["frames_30fps"][4]) / FPS for q in qs)
        print(f"  seconds from cut to step-result frame: median {gap[len(gap)//2]:.1f}, 90% {gap[9*len(gap)//10]:.1f}")
    print(f"{split}: {len(rows)} C10119 segments in {len(by_seq)} recordings -> {len(qs)} questions "
          f"(skipped {dict(skip)}); keeping {min(cap, len(qs))}")
    c = collections.Counter(q["answer"] for q in qs[:cap]); print(f"  answer letters: {dict(sorted(c.items()))}")
    return qs[:cap]


os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
for split, cap in (("train", a.train_max), ("validation", a.val_max), ("test", a.test_max)):
    if cap <= 0:
        continue
    qs = build(split, cap, rng)
    json.dump(qs, open(os.path.join(a.out, f"{split}.json"), "w"))
    frames = collections.defaultdict(dict)
    for q in qs:
        frames[q["seq"]][q["id"]] = q["frames_30fps"]
    json.dump(frames, open(os.path.join(a.out, f"{split}_frames.json"), "w"))
    if qs:
        print("  example:", json.dumps({k: qs[0][k] for k in ("id", "options", "answer", "running", "frames_30fps")})[:600])
