"""Print real examples: one FutureBench question per type, and training samples with their time gaps.
usage: python 11_show_examples.py [n_per_type] [n_train]"""
import json, random, sys, textwrap
n_fb = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 2
random.seed(1)
W = lambda s, ind="    ": textwrap.fill(str(s), 110, initial_indent=ind, subsequent_indent=ind)

def secs(t):  # "0:04:09.165" -> 249.165
    h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)

fb = json.load(open("/workspace/data/futurebench/futurebench.json"))
names = {"hop1": "1-HOP (extrapolation)", "hop2": "2-HOP (extrapolation)",
         "hop3": "3-HOP (extrapolation)", "hop5": "HOP5 = INTERPOLATION"}
print("=" * 30, "FUTUREBENCH (TEST)", "=" * 30)
for qt in ["hop1", "hop2", "hop3", "hop5"]:
    for it in random.sample([x for x in fb if x["question_type"] == qt], n_fb):
        qa = it["qa"]
        print(f"\n### {names[qt]}   source={it['data_source']}   video={it['video_path']}")
        print("  QUESTION:"); print(W(qa["Question"]))
        for k in "ABCD":
            mark = "  <-- correct" if k == qa["Answer"] else ""
            print(W(f"{k}. {qa['Options'][k]}{mark}", "    "))
        exp = qa.get("Explanation", {})
        if isinstance(exp, dict):
            print("  WHY CORRECT:"); print(W(exp.get("Correct Answer", "")))
            print("  WHY OTHERS WRONG:"); print(W(exp.get("Distractors", "")))

tr = json.load(open("/workspace/data/fl1_10k/train.json"))
print("\n\n" + "=" * 30, "TRAINING (Future-L1-50K)", "=" * 30)
gaps = []
for d in tr:
    md = d.get("meta_data", {})
    try:
        dur = secs(md["end"]) - secs(md["start"])
        gaps.append((max(d["reasoning_image"]) - d["image"][0]) * dur / 7)
    except Exception:
        pass
for d in random.sample(tr, n_tr):
    md = d["meta_data"]; dur = secs(md["end"]) - secs(md["start"]); step = dur / 7
    q = d["image"][0]
    print(f"\n### id={d['id']}   clip {md['start']} -> {md['end']} ({dur:.1f} s), frames every {step:.1f} s")
    print(f"  MODEL SEES: f{q}.jpg  (t = {(q-1)*step:.1f} s)")
    print("  FUTURE FRAMES:", ", ".join(f"f{r}.jpg (t = {(r-1)*step:.1f} s, +{(r-q)*step:.1f} s)" for r in d["reasoning_image"]))
    conv = d["conversations"]
    print("  QUESTION:"); print(W(conv[0]["value"].replace("\n", " ")))
    print("  REASONING TEXT (row C learns this; <image> = where the future frame goes):"); print(W(conv[1]["value"]))
    print("  ANSWER (row B learns only this):"); print(W(d["answer"]))
    print(f"  frames on disk: {d['video']}")
if gaps:
    gaps.sort(); n = len(gaps)
    print(f"\n\nFUTURE GAP over {n} training samples (question frame -> furthest future frame):")
    print(f"  min {gaps[0]:.1f} s | 25% {gaps[n//4]:.1f} s | median {gaps[n//2]:.1f} s | 75% {gaps[3*n//4]:.1f} s | max {gaps[-1]:.1f} s")
