"""Paired comparison of two FutureBench runs on the same 1,056 questions (lenient scoring, same as 10_rescore.py).
Prints both-right / only-X / only-Y / both-wrong and an exact McNemar p-value.
usage: python 17_paired.py Bp=/workspace/outputs/eval_rowBctx4_futurebench_future_l1 D=/workspace/outputs/eval_rowDctx4_..._s1_futurebench_future_l1"""
import json, re, sys
from math import comb
from pathlib import Path

TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
L = re.compile(r"\b([ABCD])\b")

def correct(path):
    f = sorted(Path(path).rglob("*samples*.jsonl"))[0]
    out = {}
    for line in open(f):
        d = json.loads(line); r = d.get("resps") or d.get("filtered_resps")
        while isinstance(r, list) and r: r = r[0]
        r = str(r); m = TAG.search(r)
        p = (L.search(m.group(1)).group(1) if m and L.search(m.group(1)) else "") or (L.search(r).group(1) if L.search(r) else "")
        out[d["doc_id"]] = p == d["doc"]["qa"]["Answer"].strip().upper()
    return out

(na, pa), (nb, pb) = (a.split("=", 1) for a in sys.argv[1:3])
A, B = correct(pa), correct(pb)
ids = sorted(set(A) & set(B))
both = sum(A[i] and B[i] for i in ids); oa = sum(A[i] and not B[i] for i in ids)
ob = sum(B[i] and not A[i] for i in ids); none = len(ids) - both - oa - ob
k, n = min(oa, ob), oa + ob
p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
print(f"n={len(ids)}  {na}={100*sum(A[i] for i in ids)/len(ids):.1f}%  {nb}={100*sum(B[i] for i in ids)/len(ids):.1f}%")
print(f"both right {both} | only {na} {oa} | only {nb} {ob} | both wrong {none}")
print(f"McNemar exact p = {p:.4g}  ({'significant' if p < 0.05 else 'not significant'} at 0.05)")
