"""Rescore saved FutureBench samples without rerunning the model.
For each *samples*.jsonl under the given folders, report:
  tag_rate   - % of outputs containing <answer>...</answer>
  strict     - accuracy using only a letter inside <answer> tags (lmms-eval futurebench_future_l1 rule)
  lenient    - letter inside tags if present, else a standalone A-D in the raw output
plus per-question-type accuracy for the lenient score.
usage: python 10_rescore.py /workspace/outputs/eval_rowA_futurebench /workspace/outputs/eval_rowA_futurebench_future_l1"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
LETTER = re.compile(r"\b([ABCD])\b")          # standalone letter, so "Based" does not count as B

def raw_text(d):
    r = d.get("resps") or d.get("filtered_resps")
    while isinstance(r, list) and r:
        r = r[0]
    return str(r or "")

def gt_of(d):
    doc = d.get("doc", {})
    return (doc.get("qa", {}) or {}).get("Answer", d.get("target", "")).strip().upper(), doc.get("question_type", "?")

for folder in sys.argv[1:]:
    for f in sorted(Path(folder).rglob("*samples*.jsonl")):
        n = tags = strict = lenient = 0
        by_type = defaultdict(lambda: [0, 0])
        examples = []
        for line in open(f):
            d = json.loads(line); n += 1
            out = raw_text(d); gt, qt = gt_of(d)
            m = TAG.search(out)
            s = LETTER.search(m.group(1)).group(1) if m and LETTER.search(m.group(1)) else ""
            l = s or (LETTER.search(out).group(1) if LETTER.search(out) else "")
            tags += bool(m); strict += (s == gt); lenient += (l == gt)
            by_type[qt][0] += (l == gt); by_type[qt][1] += 1
            if len(examples) < 3:
                examples.append(repr(out[:120]))
        print(f"\n== {f}")
        print(f"   n={n}  tag_rate={100*tags/n:.1f}%  strict={100*strict/n:.1f}%  lenient={100*lenient/n:.1f}%")
        print("   lenient by type:", {k: f"{100*c/t:.1f}% (n={t})" for k, (c, t) in sorted(by_type.items())})
        print("   example outputs:", *examples, sep="\n     ")
