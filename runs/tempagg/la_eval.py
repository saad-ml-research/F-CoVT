"""Score saved TempAgg validation scores (patch_save_scores.py) with our metrics, with and without post-hoc logit
adjustment (LA: score - tau * log train class frequency, priors from train.csv rows of camera C10119).
TempAgg was trained on all of train, so tau cannot be picked on held-out train as for our models; we report the
whole tau grid and TempAgg's BEST tau on validation (an upper bound that favours TempAgg).
usage: python la_eval.py scores.npz /workspace/data/a101_anticip/data/CSVs
"""
import collections, csv, os, sys
import numpy as np

z = np.load(sys.argv[1]); csvs = sys.argv[2]
cnt = {"action": collections.Counter(), "verb": collections.Counter(), "object": collections.Counter()}
with open(os.path.join(csvs, "train.csv")) as f:
    for r in csv.DictReader(f):
        if r["video"].endswith("/C10119_rgb.mp4"):
            cnt["action"][int(r["action_id"])] += 1; cnt["verb"][int(r["verb_id"])] += 1; cnt["object"][int(r["noun_id"])] += 1


def logprior(k, n):
    c = np.array([cnt[k][i] + 1 for i in range(n)], dtype=np.float64)
    return np.log(c / c.sum())


def cm_recall(sc, y):
    top = np.argsort(-sc, 1)[:, :5]
    hit = (top == y[:, None]).any(1)
    per = collections.defaultdict(list)
    for c, h in zip(y, hit):
        per[c].append(h)
    return 100 * np.mean([np.mean(v) for v in per.values()])


def all_scores(tau):
    out = {}
    for k in ("verb", "object", "action"):
        sc = z[k].astype(np.float64)
        sc = sc - sc.max(1, keepdims=True); sc = sc - np.log(np.exp(sc).sum(1, keepdims=True))   # log-softmax
        sc = sc - tau * logprior(k, sc.shape[1])
        y = z[f"{k}_labels"].astype(int)
        out[k] = cm_recall(sc, y)
        top = np.argsort(-sc, 1)[:, :5]
        out[k + "_top1"] = 100 * (top[:, 0] == y).mean(); out[k + "_top5"] = 100 * (top == y[:, None]).any(1).mean()
    return out


print(f"{len(z['ids'])} validation segments")
print(f"{'tau':>5} | verb  object  action | action top-1  top-5 acc")
best = None
for tau in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3):
    r = all_scores(tau)
    print(f"{tau:5.2f} | {r['verb']:5.1f}  {r['object']:5.1f}  {r['action']:5.1f}  | {r['action_top1']:5.1f}  {r['action_top5']:5.1f}")
    sel = (r["verb"] + r["object"] + r["action"]) / 3
    if best is None or sel > best[0]:
        best = (sel, tau, r)
print(f"best tau on validation (favours TempAgg): {best[1]} -> verb {best[2]['verb']:.1f} object {best[2]['object']:.1f} "
      f"action {best[2]['action']:.1f} top-1 {best[2]['action_top1']:.1f} top-5 acc {best[2]['action_top5']:.1f}")
