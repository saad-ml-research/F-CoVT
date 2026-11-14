"""Model A v2 probe on the V-JEPA 2 token bank (01_bank.py), in the style of V-JEPA 2's anticipation probe:
attentive probe = token projection + time/space/type embeddings -> self-attention blocks -> cross-attention from
3 learned queries (action / verb / object) -> 3 linear classifiers, focal loss (gamma 2) summed over the three.
Rows (what the probe reads; every row sees the same past):
  enc        past encoder tokens only (slots 0-7: the 2 s before the cut)
  enc+pred   + the frozen pretrained V-JEPA 2 predictor's tokens for the future        <- honest imagination
  pred       predictor tokens only (no past)                                           (as V-JEPA 2's table)
  enc+real   + the REAL future tokens (oracle, cheating ceiling)
--future all : future slots 8-15 (0 .. +2 s);  t3t4 : slot 13 (+1.5 s, in the next action) and the slot of the end
               of the next action (our best targets from the frame-split study).
Model selection on a held-out 10% of train by the mean of verb / object / action class-mean top-5 recall (the
reported metric); validation is only scored. Mean +- sd over seeds.
usage: python 02_probe.py /workspace/outputs/vjepa_bank [--rows enc,enc+pred,pred,enc+real] [--future all|t3t4]
"""
import argparse, collections, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("bank")
ap.add_argument("--rows", default="enc,enc+pred,pred,enc+real")
ap.add_argument("--future", default="all", choices=["all", "t3t4"])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"
SIZES = [1400, 32, 100]

tr, va = (torch.load(os.path.join(a.bank, f"{s}.pt")) for s in ("train", "validation"))
FS0 = tr["first_future_slot"]                                    # 8
NPS, NFS, NSP = tr["past"].shape[1], tr["pred"].shape[1], tr["past"].shape[2]   # 8 past slots, 8 future, 16 regions
D = tr["past"].shape[-1]
print(f"train {len(tr['ids'])}  val {len(va['ids'])} | past {tuple(tr['past'].shape[1:])} | future {a.future}", flush=True)


def fut_slots(d):                                                # future slot indices (0-based within the future)
    n = len(d["ids"])
    if a.future == "all":
        return torch.arange(NFS).expand(n, -1)
    t3 = torch.full((n,), 13 - FS0)
    return torch.stack([t3, d["t4_slot"] - FS0], 1)


def build(d, row):
    """tokens [N, L, D] fp16, time index [N, L], type index [N, L] (0 past, 1 predicted, 2 real), space index [N, L]."""
    n = len(d["ids"])
    parts, times, types = [], [], []
    if row in ("enc", "enc+pred", "enc+real"):
        parts.append(d["past"]); times.append(torch.arange(NPS).expand(n, -1)); types.append(0)
    fs = fut_slots(d)
    for key, ty in (("pred", 1), ("real", 2)):
        if key in row.split("+") or row == key:
            parts.append(torch.gather(d[key], 1, fs[:, :, None, None].expand(-1, -1, NSP, D)))
            times.append(fs + FS0); types.append(ty)
    tok = torch.cat(parts, 1)                                    # [N, S, 16, D]
    tim = torch.cat(times, 1)                                    # [N, S]
    typ = torch.cat([torch.full_like(t, ty) for t, ty in zip(times, types)], 1)
    S = tok.shape[1]
    tok = tok.reshape(n, S * NSP, D)
    tim = tim[:, :, None].expand(-1, -1, NSP).reshape(n, -1)
    typ = typ[:, :, None].expand(-1, -1, NSP).reshape(n, -1)
    spc = torch.arange(NSP).repeat(S).expand(n, -1)
    return tok.to(dev), tim.to(dev), typ.to(dev), spc.to(dev)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        d = a.d
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.time = nn.Embedding(NPS + NFS, d); self.kind = nn.Embedding(3, d); self.space = nn.Embedding(NSP, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, a.blocks)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(1, 3, d) * 0.02)       # action, verb, object queries
        self.xattn = nn.MultiheadAttention(d, 8, dropout=a.drop, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        self.drop = nn.Dropout(a.drop)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def forward(self, tok, tim, typ, spc):
        x = self.inp(tok.float()) + self.time(tim) + self.kind(typ) + self.space(spc)
        x = self.norm(self.blocks(x))
        q, _ = self.xattn(self.q.expand(len(x), -1, -1), x, x)
        q = self.drop(self.qnorm(q))
        return [self.outs[j](q[:, j]) for j in range(3)]


def focal(logits, y):
    logp = F.log_softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** a.gamma * logp).mean()


def cm_recall(logits, y):
    hit = (logits.topk(5, -1).indices == y[:, None]).any(-1)
    per = collections.defaultdict(list)
    for c, h in zip(y.tolist(), hit.tolist()):
        per[c].append(h)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def scores(lg, y):
    return {"action": cm_recall(lg[0], y[:, 0]), "verb": cm_recall(lg[1], y[:, 1]), "noun": cm_recall(lg[2], y[:, 2]),
            "action_top1": 100 * (lg[0].argmax(-1) == y[:, 0]).float().mean().item()}


@torch.no_grad()
def predict(m, X, idx=None):
    m.eval()
    idx = torch.arange(len(X[0]), device=dev) if idx is None else idx
    outs = [[], [], []]
    for s in range(0, len(idx), 512):
        b = idx[s:s + 512]
        for j, o in enumerate(m(*(t[b] for t in X))):
            outs[j].append(o.float())
    return [torch.cat(o) for o in outs]


Ytr, Yva = tr["y"].to(dev), va["y"].to(dev)
all_res = {}
for row in a.rows.split(","):
    Xtr, Xva = build(tr, row), build(va, row)
    rs = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        perm = torch.randperm(len(Ytr), generator=torch.Generator().manual_seed(seed)).to(dev)
        ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
        m = Probe().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * math.ceil(len(fi) / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        best = (-1.0, None, 0)
        for ep in range(a.epochs):
            m.train()
            p = fi[torch.randperm(len(fi), device=dev)]
            tot, nb = 0.0, 0
            for s in range(0, len(p), a.bs):
                b = p[s:s + a.bs]
                lg = m(*(t[b] for t in Xtr))
                loss = sum(focal(lg[j], Ytr[b, j]) for j in range(3))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item(); nb += 1
            h = scores(predict(m, Xtr, ho), Ytr[ho])
            sel = (h["verb"] + h["noun"] + h["action"]) / 3
            if sel > best[0]:
                best = (sel, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1)
            if a.every and ((ep + 1) % a.every == 0 or ep == 0):
                v = scores(predict(m, Xva), Yva)
                print(f"  [{row} seed {seed}] epoch {ep + 1:3d}  train focal {tot / nb:.3f}  held-out mean recall {sel:5.1f}"
                      f"{' *' if best[2] == ep + 1 else '  '}  val verb {v['verb']:5.1f}  object {v['noun']:5.1f}"
                      f"  action {v['action']:4.1f}", flush=True)
        m.load_state_dict(best[1])
        r = scores(predict(m, Xva), Yva); r["epoch"] = best[2]
        rs.append(r)
    all_res[row] = rs
    ms = lambda k: (lambda t: f"{t.mean():5.1f} +- {t.std():3.1f}")(torch.tensor([r[k] for r in rs]))   # noqa: E731
    print(f"{row:9s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}"
          f" | epochs {[r['epoch'] for r in rs]}", flush=True)
    del Xtr, Xva; torch.cuda.empty_cache()

json.dump({"args": vars(a), "results": all_res},
          open(os.path.join(a.bank, f"probe_{a.future}{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)
if "enc" in all_res and "enc+pred" in all_res:
    g = torch.tensor([p["verb"] - e["verb"] for p, e in zip(all_res["enc+pred"], all_res["enc"])])
    print(f"\nGate 2, verb gain enc+pred - enc: {g.mean():+.1f} +- {g.std():.1f}  (pass: >= +2 and larger than the spread)")
