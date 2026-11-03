"""Model A: JEPA-style future prediction for action anticipation, on the frozen DINOv2 bank (09_dino_bank.py).
  past frames (default P3: every 0.25 s over the last 1.75 s, 8 frames) -> frozen DINOv2 z_t
  temporal transformer (trainable) -> state s
  predictor (trainable) -> guessed future features zhat for each target (default T3 = +1.5 s in the next action,
                           T4 = end of the next action), predicted as a CHANGE from the current frame: zhat = z_now + delta
  action head reads [s ; proj(zhat)] -> action / verb / object
  loss = CE + lambda * |zhat - z_future|^2   (z_future = DINOv2 of the real future frames; training only)
Variants (all with the same transformer, head and budget):
  base    head reads s only, no predictor
  jepa    head reads s + zhat, JEPA loss on                         <- ours
  ctrl    head reads s + zhat, JEPA loss off (lambda 0)              extra layers without the future
  oracle  head reads s + the REAL future features                    ceiling (cheating)
Reported on validation: class-mean top-5 recall (verb / object / action), action top-1, and "explained"
(1 - |zhat - z|^2 / |z_now - z|^2: share of the change from now to the target the predictor recovers).
Epoch picked on a held-out 10% of train (CE loss); mean +- sd over seeds.
usage: python 11_jepa.py /workspace/outputs/dino_bank [--variants base,jepa,ctrl,oracle] [--seeds 3] [--lam 1.0]
"""
import argparse, collections, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("bank")
ap.add_argument("--variants", default="base,jepa,ctrl,oracle")
ap.add_argument("--past", default="-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0")
ap.add_argument("--targets", default="+1.50s,next_end")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--detach", action="store_true", help="head reads zhat.detach(): only the JEPA loss shapes the guess")
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--tag", default="")
ap.add_argument("--every", type=int, default=5, help="print training progress every N epochs (0 = quiet)")
a = ap.parse_args()
dev = "cuda"
SIZES = [1400, 32, 100]

tr, va = (torch.load(os.path.join(a.bank, f"{s}.pt")) for s in ("train", "validation"))
names = tr["names"]
col = {n: i for i, n in enumerate(names)}
pc = [col[f"{float(t):+.2f}s"] for t in a.past.split(",")]
tc = [col[t] for t in a.targets.split(",")]
now = col["+0.00s"]
assert now in pc, "the past must include the current frame (0.0)"
mu = tr["feats"][:, pc].float().mean((0, 1)); sd = tr["feats"][:, pc].float().std((0, 1)) + 1e-4


def prep(d):
    f = (d["feats"].float() - mu) / sd
    return f[:, pc].to(dev), f[:, tc].to(dev), f[:, now].to(dev), d["y"].to(dev)


Ptr, Ttr, Ntr, Ytr = prep(tr)
Pva, Tva, Nva, Yva = prep(va)
D = Ptr.shape[-1]
print(f"train {len(Ptr)}  val {len(Pva)} | past {len(pc)} frames {a.past} | targets {a.targets} | lambda {a.lam}"
      f"{' detach' if a.detach else ''}", flush=True)


class ModelA(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.v = variant
        d, nt = a.d, len(tc)
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Parameter(torch.zeros(1, len(pc) + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, a.layers)
        self.norm = nn.LayerNorm(d)
        if variant in ("jepa", "ctrl"):
            self.pred = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, nt * D))
            nn.init.zeros_(self.pred[-1].weight); nn.init.zeros_(self.pred[-1].bias)   # starts as "copy now"
        if variant != "base":
            self.fproj = nn.Sequential(nn.LayerNorm(nt * D), nn.Linear(nt * D, d))
        hin = d if variant == "base" else 2 * d
        self.drop = nn.Dropout(a.drop)
        self.outs = nn.ModuleList(nn.Linear(hin, n) for n in SIZES)

    def forward(self, past, znow, zreal=None):
        x = torch.cat([self.cls.expand(len(past), -1, -1), self.inp(past)], 1) + self.pos
        s = self.norm(self.enc(x)[:, 0])
        zhat = None
        if self.v == "base":
            h = s
        else:
            if self.v == "oracle":
                fut = zreal
            else:
                zhat = znow[:, None] + self.pred(s).view(len(s), len(tc), D)
                fut = zhat.detach() if a.detach else zhat
            h = torch.cat([s, self.fproj(fut.flatten(1))], 1)
        h = self.drop(h)
        return [o(h) for o in self.outs], zhat


def ce(lg, y):
    return sum(F.cross_entropy(lg[j], y[:, j]) for j in range(3))


def cm_recall(logits, y):
    hit = (logits.topk(5, -1).indices == y[:, None]).any(-1)
    per = collections.defaultdict(list)
    for c, h in zip(y.tolist(), hit.tolist()):
        per[c].append(h)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def run(variant, seed):
    torch.manual_seed(seed)
    perm = torch.randperm(len(Ptr), generator=torch.Generator().manual_seed(seed)).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    m = ModelA(variant).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * math.ceil(len(fi) / a.bs), pct_start=0.05)
    lam = a.lam if variant == "jepa" else 0.0
    best = (math.inf, None, -1)
    for ep in range(a.epochs):
        m.train()
        p = fi[torch.randperm(len(fi), device=dev)]
        sums, nb = [0.0, 0.0], 0
        for s in range(0, len(p), a.bs):
            b = p[s:s + a.bs]
            lg, zhat = m(Ptr[b], Ntr[b], Ttr[b])
            loss = ce(lg, Ytr[b])
            sums[0] += loss.item(); nb += 1
            if zhat is not None:
                jl = F.mse_loss(zhat, Ttr[b]); sums[1] += jl.item()
                if lam:
                    loss = loss + lam * jl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            hl = ce(m(Ptr[ho], Ntr[ho], Ttr[ho])[0], Ytr[ho]).item()
        if hl < best[0]:
            best = (hl, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1)
        if a.every and ((ep + 1) % a.every == 0 or ep == 0):
            with torch.no_grad():                    # validation shown for watching only; never used for choices
                lgv, zv = m(Pva, Nva, Tva)
            msg = (f"  [{variant} seed {seed}] epoch {ep + 1:3d}  train CE {sums[0] / nb:6.2f}"
                   + (f"  JEPA {sums[1] / nb:.3f}{'' if lam else ' (not trained)'}" if zhat is not None else "")
                   + f"  held-out CE {hl:6.2f}{' *' if best[2] == ep + 1 else '  '}"
                   + f"  val verb {cm_recall(lgv[1], Yva[:, 1]):5.1f}  object {cm_recall(lgv[2], Yva[:, 2]):5.1f}")
            if zv is not None:
                msg += "  explained " + " ".join(
                    f"{1 - ((zv[:, j] - Tva[:, j]) ** 2).sum().item() / ((Nva - Tva[:, j]) ** 2).sum().item():+.3f}"
                    for j in range(len(tc)))
            print(msg, flush=True)
    m.load_state_dict(best[1]); m.eval()
    with torch.no_grad():
        lg, zhat = m(Pva, Nva, Tva)
    r = {"verb": cm_recall(lg[1], Yva[:, 1]), "noun": cm_recall(lg[2], Yva[:, 2]), "action": cm_recall(lg[0], Yva[:, 0]),
         "action_top1": 100 * (lg[0].argmax(-1) == Yva[:, 0]).float().mean().item(), "epoch": best[2]}
    if zhat is not None:
        for j, t in enumerate(a.targets.split(",")):
            r[f"explained_{t}"] = 1 - ((zhat[:, j] - Tva[:, j]) ** 2).sum().item() / ((Nva - Tva[:, j]) ** 2).sum().item()
    return r


all_res = {}
for v in a.variants.split(","):
    rs = [run(v, s) for s in range(a.seeds)]
    all_res[v] = rs
    def ms(k):
        t = torch.tensor([r[k] for r in rs]); return f"{t.mean():5.1f} +- {t.std():3.1f}"
    ex = "".join(f"  explained {k[10:]} {torch.tensor([r[k] for r in rs]).mean():+.3f}" for k in rs[0] if k.startswith("explained_"))
    print(f"{v:7s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}"
          f" | epochs {[r['epoch'] for r in rs]}{ex}", flush=True)

out = os.path.join(a.bank, f"jepa{('_' + a.tag) if a.tag else ''}.json")
json.dump({"args": vars(a), "results": all_res}, open(out, "w"), indent=1)
if "base" in all_res and "jepa" in all_res:
    g = torch.tensor([j["verb"] - b["verb"] for j, b in zip(all_res["jepa"], all_res["base"])])
    print(f"\nGate 1, verb gain jepa - base: {g.mean():+.1f} +- {g.std():.1f}  (pass: >= +2 and larger than the spread; "
          f"explained > 0; and jepa > ctrl)")
