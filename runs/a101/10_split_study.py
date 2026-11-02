"""Frame-split study on the DINOv2 bank (09_dino_bank.py): 5 past inputs x 5 future targets = 25 combinations.
For each combination, on validation:
  oracle gain   probe(past + REAL target)      - probe(past)    how much knowing the future would help
  pred gain     probe(past + PREDICTED target) - probe(past)    how much imagining it actually helps
  explained     1 - |pred - real|^2 / |now - real|^2             how much of the change from now to the target the
                                                                 predictor recovers (> 0: better than copying now)
Predictor: ridge regression from the concatenated past features to the target features (lambda picked on a
held-out part of train); train-set predictions are 5-fold out-of-fold so the probe does not learn to over-trust
them. Probe: the same linear probe as 07_probe.py (held-out epoch selection). Main number: class-mean top-5
recall (official Assembly101 metric) for verb, object, action.
usage: python 10_split_study.py /workspace/outputs/dino_bank [--epochs 40] [--seed 0]
"""
import argparse, collections, json, math, os, sys
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("bank")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
SIZES = [1400, 32, 100]

PAST = {
    "P1 f5 only": [0.0],
    "P2 5 x 1 s": [-4.0, -3.0, -2.0, -1.0, 0.0],
    "P3 8 x 0.25 s": [-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0],
    "P4 8 x 1 s": [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
    "P5 long + dense": [-8.0, -6.0, -4.0, -2.0, -1.5, -1.0, -0.5, 0.0],
}
TARGET = {
    "T1 +0.5 s": "+0.50s",
    "T2 +1.0 s (onset)": "+1.00s",
    "T3 +1.5 s (in action)": "+1.50s",
    "T4 next action end": "next_end",
    "T5 step end (f7)": "step_end",
}

tr, va = (torch.load(os.path.join(a.bank, f"{s}.pt")) for s in ("train", "validation"))
names = tr["names"]
col = {n: i for i, n in enumerate(names)}
col.update({f"{float(n[:-1]):+.2f}s": i for i, n in enumerate(names) if n.endswith("s") and n[0] in "+-"})
Ftr, Fva = tr["feats"].float(), va["feats"].float()
mu, sd = Ftr.mean((0, 1)), Ftr.std((0, 1)) + 1e-4                      # one standardization for every frame
Ftr, Fva = ((Ftr - mu) / sd).to(dev), ((Fva - mu) / sd).to(dev)
Ytr, Yva = tr["y"].to(dev), va["y"].to(dev)
NOW = col["+0.00s"]
print(f"train {len(Ftr)}  val {len(Fva)}  frames per question {len(names)}  feature dim {Ftr.shape[-1]}", flush=True)


def cols(times):
    return [col[f"{t:+.2f}s"] for t in times]


# ----------------------------------------------------------------------------- ridge predictor
def ridge_fit(X, Y, lam):
    D = X.shape[1]
    G = X.T @ X + lam * len(X) / D * torch.eye(D, device=dev)
    return torch.linalg.solve(G, X.T @ Y)


def predict_all(Xtr, Ytr_, Xva):
    """lambda on a 10% held-out part of train, then out-of-fold train predictions + full-train val predictions."""
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(Xtr), generator=g).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    best = None
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
        W = ridge_fit(Xtr[fi], Ytr_[fi], lam)
        err = ((Xtr[ho] @ W - Ytr_[ho]) ** 2).mean().item()
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    oof = torch.empty_like(Ytr_)
    for k in range(5):
        te = perm[k::5]
        trn = torch.cat([perm[j::5] for j in range(5) if j != k])
        oof[te] = Xtr[te] @ ridge_fit(Xtr[trn], Ytr_[trn], lam)
    return oof, Xva @ ridge_fit(Xtr, Ytr_, lam), lam


# ----------------------------------------------------------------------------- linear probe (as 07_probe.py)
class Head(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.outs = torch.nn.ModuleList(torch.nn.Linear(d, n) for n in SIZES)

    def forward(self, x):
        x = self.norm(x)
        return [o(x) for o in self.outs]


def lossf(lg, y):
    return sum(F.cross_entropy(lg[j], y[:, j]) for j in range(3))


def cm_recall(logits, y):
    top5 = logits.topk(5, -1).indices
    hit = (top5 == y[:, None]).any(-1)
    per = collections.defaultdict(list)
    for c, h in zip(y.tolist(), hit.tolist()):
        per[c].append(h)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def probe(Xtr, Xva):
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(Xtr), generator=g).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    head = Head(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = (math.inf, None)
    for ep in range(a.epochs):
        head.train()
        p = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(p), 256):
            b = p[s:s + 256]
            l = lossf(head(Xtr[b]), Ytr[b]); opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            hl = lossf(head(Xtr[ho]), Ytr[ho]).item()
        if hl < best[0]:
            best = (hl, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        lg = head(Xva)
    return {"verb": cm_recall(lg[1], Yva[:, 1]), "noun": cm_recall(lg[2], Yva[:, 2]),
            "action": cm_recall(lg[0], Yva[:, 0]),
            "action_top1": 100 * (lg[0].argmax(-1) == Yva[:, 0]).float().mean().item()}


# ----------------------------------------------------------------------------- the grid
results = {}
for pn, times in PAST.items():
    c = cols(times)
    Xtr, Xva = Ftr[:, c].flatten(1), Fva[:, c].flatten(1)
    base = probe(Xtr, Xva)
    results[pn] = {"base": base, "targets": {}}
    print(f"\n{pn}  ({len(c)} frames)  base: verb {base['verb']:.1f}  object {base['noun']:.1f}  "
          f"action {base['action']:.1f}  action top-1 {base['action_top1']:.1f}", flush=True)
    for tn, key in TARGET.items():
        t = col[key]
        Ttr, Tva = Ftr[:, t], Fva[:, t]
        oof, pva, lam = predict_all(Xtr, Ttr, Xva)
        explained = 1 - ((pva - Tva) ** 2).sum().item() / ((Fva[:, NOW] - Tva) ** 2).sum().item()
        cos_pred = F.cosine_similarity(pva, Tva, -1).mean().item()
        cos_copy = F.cosine_similarity(Fva[:, NOW], Tva, -1).mean().item()
        orc = probe(torch.cat([Xtr, Ttr], 1), torch.cat([Xva, Tva], 1))
        prd = probe(torch.cat([Xtr, oof], 1), torch.cat([Xva, pva], 1))
        results[pn]["targets"][tn] = {"oracle": orc, "pred": prd, "explained": explained, "lambda": lam,
                                      "cos_pred": cos_pred, "cos_copy_now": cos_copy}
        d = lambda r, k: r[k] - base[k]                                    # noqa: E731
        print(f"  {tn:24s} oracle gain verb {d(orc,'verb'):+5.1f} obj {d(orc,'noun'):+5.1f} act {d(orc,'action'):+4.1f}"
              f" | predicted gain verb {d(prd,'verb'):+5.1f} obj {d(prd,'noun'):+5.1f} act {d(prd,'action'):+4.1f}"
              f" | explained {explained:+.2f} (cos pred {cos_pred:.2f} vs copy-now {cos_copy:.2f})", flush=True)

json.dump(results, open(os.path.join(a.bank, f"split_study_seed{a.seed}.json"), "w"), indent=1)

print("\nVERB class-mean top-5 recall gain over the past-only probe: oracle / predicted")
print(f"{'':18s}" + "".join(f"{tn.split(' (')[0]:>16s}" for tn in TARGET))
for pn, r in results.items():
    print(f"{pn:18s}" + "".join(f"{r['targets'][tn]['oracle']['verb']-r['base']['verb']:+7.1f} /{r['targets'][tn]['pred']['verb']-r['base']['verb']:+5.1f} "
                                for tn in TARGET))
print("\nexplained (share of the now -> target change the predictor recovers; > 0 beats copying now)")
print(f"{'':18s}" + "".join(f"{tn.split(' (')[0]:>16s}" for tn in TARGET))
for pn, r in results.items():
    print(f"{pn:18s}" + "".join(f"{r['targets'][tn]['explained']:16.2f}" for tn in TARGET))
