"""Late fusion of TempAgg (patch_save_scores.py .npz) with our probe models (04_probe.py --save_val), matched by
segment id, + seed ensembles of ours. Fusion = sum of log-probs with weight w on ours (w picked on a fixed grid and
reported for every w; no tuning on validation is used to choose the reported 'equal weight' row).
Also prints the LA curve (tau 0..3) for the equal-weight fusion, to compare at matched top-5 accuracy.
usage: python fuse_eval.py tempagg.npz VAL_DIR /workspace/data/a101_vlm/validation.json CSVs --rows fuse_kd,fuse_kd+hS --seeds 5
"""
import argparse, collections, csv, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("val_dir"); ap.add_argument("val_index"); ap.add_argument("csvs")
ap.add_argument("--rows", default="fuse_kd"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--anno", default="/workspace/data/a101_anticip", help="searched for tail_validation_segments.txt / validation_split_seq.txt")
a = ap.parse_args()
z = np.load(a.npz)
order = {int(q["id"]): i for i, q in enumerate(json.load(open(a.val_index)))}           # our validation row order
sel = np.array([order[int(i)] for i in z["ids"]])                                           # TempAgg rows -> our rows
Y = {"action": z["action_labels"].astype(int), "verb": z["verb_labels"].astype(int), "object": z["object_labels"].astype(int)}
cnt = {k: collections.Counter() for k in Y}
with open(os.path.join(a.csvs, "train.csv")) as f:
    for r in csv.DictReader(f):
        if r["video"].endswith("/C10119_rgb.mp4"):
            cnt["action"][int(r["action_id"])] += 1; cnt["verb"][int(r["verb_id"])] += 1; cnt["object"][int(r["noun_id"])] += 1


def lsm(x):
    x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))


def score(S, tau=0.0):
    out = {}
    for k in ("verb", "object", "action"):
        n = S[k].shape[1]
        lp = np.log(np.array([cnt[k][i] + 1 for i in range(n)], float)); lp -= np.log(lp.size and np.exp(lp).sum())
        sc = S[k] - tau * lp
        top = np.argsort(-sc, 1)[:, :5]; y = Y[k]
        per = collections.defaultdict(list)
        for c, h in zip(y, (top == y[:, None]).any(1)):
            per[c].append(h)
        out[k] = 100 * np.mean([np.mean(v) for v in per.values()])
        out[k + "_top1"] = 100 * (top[:, 0] == y).mean(); out[k + "_top5"] = 100 * (top == y[:, None]).any(1).mean()
    return out


TA = {k: lsm(z[k].astype(np.float64)) for k in ("verb", "object", "action")}

# ------------------------------------------------------------------ official splits: head / tail segments, seen / unseen toys
def find(name):
    for root, _, files in os.walk(a.anno):
        if name in files:
            return os.path.join(root, name)


ids = z["ids"].astype(int)
SPLITS = {"overall": np.ones(len(ids), bool)}
tf = find("tail_validation_segments.txt")
if tf:
    tail = {int(x.strip()) for x in open(tf) if x.strip()}
    SPLITS["head"], SPLITS["tail"] = ~np.isin(ids, list(tail)), np.isin(ids, list(tail))
else:
    print("tail_validation_segments.txt not found: no head / tail split")
vid_of, shared_of = {}, {}
with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
    for r in csv.DictReader(f):
        vid_of[int(r["id"])] = r["video"].split("/")[0]
        if "shared" in r:
            shared_of[int(r["id"])] = r["shared"].strip()
sf = find("validation_split_seq.txt")
if sf:
    unseen = {l.split("\t")[0].strip() for l in open(sf) if len(l.split("\t")) > 1 and l.split("\t")[1].strip() == "notshared"}
    un = np.array([vid_of[i] in unseen for i in ids])
elif shared_of:
    un = np.array([shared_of[i] == "notshared" for i in ids])
else:
    un = None; print("no seen / unseen information found")
if un is not None:
    SPLITS["seen"], SPLITS["unseen"] = ~un, un
print("splits:", {k: int(v.sum()) for k, v in SPLITS.items()}, "| tail file:", tf, "| seen/unseen:", sf or "csv 'shared' column")


def cm(sc, y, mask):
    top = np.argsort(-sc[mask], 1)[:, :5]; yy = y[mask]
    per = collections.defaultdict(list)
    for c, h in zip(yy, (top == yy[:, None]).any(1)):
        per[c].append(h)
    return 100 * np.mean([np.mean(v) for v in per.values()])


def breakdown(name, S, tau=0.0):
    adj = {}
    for k in ("verb", "object", "action"):
        n = S[k].shape[1]
        c = np.array([cnt[k][i] + 1 for i in range(n)], float)
        adj[k] = S[k] - tau * np.log(c / c.sum())
    cells = [f"{sp}: " + " / ".join(f"{cm(adj[k], Y[k], m):4.1f}" for k in ("verb", "object", "action")) for sp, m in SPLITS.items()]
    print(f"  {name:40s} " + " | ".join(cells))


def ours(row, seeds):
    acc = None
    for s_ in seeds:
        v = torch.load(os.path.join(a.val_dir, f"{row.replace('|', '_')}_seed{s_}.pt"))     # [action, verb, noun] log-probs
        d = {"action": v[0].float().numpy()[sel], "verb": v[1].float().numpy()[sel], "object": v[2].float().numpy()[sel]}
        acc = d if acc is None else {k: acc[k] + d[k] for k in d}
    return {k: lsm(acc[k] / len(seeds)) for k in acc}


fmt = lambda name, r: (f"{name:44s} verb {r['verb']:5.1f} object {r['object']:5.1f} action {r['action']:5.1f} | "   # noqa: E731
                       f"top-1 {r['action_top1']:5.1f} top-5 acc {r['action_top5']:5.1f}")
print(f"{len(sel)} validation segments matched")
print(fmt("TempAgg (official)", score(TA)))
BREAK = [("TempAgg (official)", TA)]
for row in a.rows.split(","):
    seeds = [s_ for s_ in range(a.seeds) if os.path.exists(os.path.join(a.val_dir, f"{row.replace('|', '_')}_seed{s_}.pt"))]
    assert seeds, f"no saved predictions for {row}"
    single = [score(ours(row, [s_])) for s_ in seeds]
    m = {k: np.mean([r[k] for r in single]) for k in single[0]}
    print(fmt(f"{row} (single seed, mean of {len(seeds)})", m))
    E = ours(row, seeds)
    print(fmt(f"{row} ({len(seeds)}-seed ensemble)", score(E)))
    BREAK.append((f"{row} ({len(seeds)}-seed ens.)", E))
    for w in (0.5, 1.0, 2.0):
        F_ = {k: lsm(TA[k] + w * E[k]) for k in TA}
        print(fmt(f"TempAgg + {w:g} x {row} ensemble", score(F_)))
    # matched-accuracy comparison: our LA curve vs TempAgg's, recall interpolated at the same top-5 accuracy
    taus = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6]
    ta_c = sorted((score(TA, t)["action_top5"], score(TA, t)["action"]) for t in taus)
    xs, ys = np.array([c[0] for c in ta_c]), np.array([c[1] for c in ta_c])
    print(f"  matched top-5 accuracy, {row} ({len(seeds)}-seed ensemble) + LA vs TempAgg + LA:")
    for t in taus[:11]:
        r = score(E, t)
        if xs.min() <= r["action_top5"] <= xs.max():
            ta = float(np.interp(r["action_top5"], xs, ys))
            print(f"    tau {t:4.2f} | top-5 acc {r['action_top5']:5.1f} | ours {r['action']:5.1f}  TempAgg {ta:5.1f}  gap {r['action'] - ta:+.1f}")
    F_ = {k: lsm(TA[k] + E[k]) for k in TA}
    BREAK.append((f"TempAgg + {row}", F_))
    print(f"  LA curve, TempAgg + {row} ensemble (equal weight):")
    for tau in (0, 0.5, 1, 1.5, 2, 2.5, 3):
        r = score(F_, tau)
        print(f"    tau {tau:4.2f} | action {r['action']:5.1f} | top-1 {r['action_top1']:5.1f} top-5 acc {r['action_top5']:5.1f}")

print("\nclass-mean recall@5 verb / object / action by split (official tail list and seen / unseen toys)")
for tau in (0.0, 1.0):
    print(f" LA tau {tau:g}:")
    for name, S in BREAK:
        breakdown(name, S, tau)
