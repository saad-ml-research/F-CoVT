"""Accuracy by video source and by question type for several eval runs (lenient scoring, same as 10_rescore.py).
usage: python 15_by_source.py A=/workspace/outputs/eval_rowA_futurebench_future_l1 B=... Bp=... C=..."""
import json, re, sys
from collections import defaultdict
from pathlib import Path

TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
L = re.compile(r"\b([ABCD])\b")

def score(f):
    by = defaultdict(lambda: [0, 0])
    for line in open(f):
        d = json.loads(line); r = d.get("resps") or d.get("filtered_resps")
        while isinstance(r, list) and r: r = r[0]
        r = str(r); m = TAG.search(r)
        p = (L.search(m.group(1)).group(1) if m and L.search(m.group(1)) else "") or (L.search(r).group(1) if L.search(r) else "")
        doc = d["doc"]; ok = p == doc["qa"]["Answer"].strip().upper()
        for k in ("src:" + doc["data_source"], "type:" + doc["question_type"], "ALL"):
            by[k][0] += ok; by[k][1] += 1
    return by

runs = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    f = sorted(Path(path).rglob("*samples*.jsonl"))[0]
    runs[name] = score(f)
keys = sorted({k for r in runs.values() for k in r}, key=lambda k: (k != "ALL", k[:4], -runs[next(iter(runs))].get(k, [0, 0])[1]))
names = list(runs)
print(f"{'':22s}{'n':>6s}" + "".join(f"{n:>8s}" for n in names))
for k in keys:
    n = next(r[k][1] for r in runs.values() if k in r)
    label = {"type:hop5": "type:interp"}.get(k, k)
    print(f"{label:22s}{n:6d}" + "".join(f"{100*r[k][0]/r[k][1]:8.1f}" if k in r else f"{'-':>8s}" for r in runs.values()))
