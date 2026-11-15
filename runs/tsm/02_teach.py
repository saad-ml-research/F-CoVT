"""Future teaching on the TSM bank (01_bank.py), official Assembly101 anticipation set, camera C10119.
Tokens: history (30 x 2 s, last 60 s), recent (8 x 0.25 s, last 2 s), future (8 x 0.25 s, next 2 s; training only).
Same attentive probe + focal loss + recipe as the V-JEPA 2 runs. Rows:
  recent   student on the recent 2 s only
  hist     student on history + recent (the past the published methods use)
  oracle   history + recent + REAL future (cheating ceiling; also the teacher)
  kd       hist student + KL to the oracle teacher (future teaching)
  kd_rec   recent-only student + KL to the oracle teacher
Validation = validation_challenge (13,494 segments, all kept). Metric: class-mean top-5 recall over the classes
present, for verb (17), object (90), action (1,063); action top-1. Epoch picked on a held-out 10% of train.
usage: python 02_teach.py /workspace/outputs/tsm_bank [--rows recent,hist,oracle,kd] [--seeds 3]
"""
import argparse, collections, copy, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("bank")
ap.add_argument("--rows", default="recent,hist,oracle,kd,kd_rec")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--lam_kd", type=float, default=1.0)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"

tr, va = (torch.load(os.path.join(a.bank, f"{s}.pt")) for s in ("train", "validation"))
SIZES = [int(max(tr["y"][:, j].max(), va["y"][:, j].max())) + 1 for j in range(3)]     # action, verb, object
D = tr["recent"].shape[-1]
mu = tr["recent"][:4000].float().mean((0, 1)); sd = tr["recent"][:4000].float().std((0, 1)) + 1e-3


def prep(d):
    return {k: d[k].to(dev) for k in ("recent", "hist", "hist_mask", "fut", "y")}


TR, VA = prep(tr), prep(va)
MU, SD = mu.to(dev), sd.to(dev)
print(f"train {len(TR['y'])}  val {len(VA['y'])} | classes action/verb/object {SIZES} | rows {a.rows}", flush=True)
PARTS = {"recent": ("recent",), "kd_rec": ("recent",), "hist": ("hist", "recent"), "kd": ("hist", "recent"),
         "oracle": ("hist", "recent", "fut")}
KIND = {"hist": 0, "recent": 1, "fut": 2}
TIME0 = {"hist": 0, "recent": 30, "fut": 38}                                     # 46 time positions in order


def tokens(D_, row, idx):
    xs, ms, tim, kd = [], [], [], []
    for p in PARTS[row]:
        x = D_[p][idx]
        n = x.shape[1]
        xs.append(x)
        ms.append(D_["hist_mask"][idx] if p == "hist" else torch.ones(len(idx), n, dtype=torch.bool, device=dev))
        tim.append(torch.arange(TIME0[p], TIME0[p] + n, device=dev)); kd.append(torch.full((n,), KIND[p], device=dev))
    x = (torch.cat(xs, 1).float() - MU) / SD
    return x, torch.cat(ms, 1), torch.cat(tim), torch.cat(kd)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        d = a.d
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.time = nn.Embedding(46, d); self.kind = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, a.blocks)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, dropout=a.drop, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        self.drop = nn.Dropout(a.drop)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def forward(self, x, mask, tim, kind):
        h = self.inp(x) + self.time(tim) + self.kind(kind)
        pad = ~mask
        h = self.norm(self.blocks(h, src_key_padding_mask=pad))
        q, _ = self.xattn(self.q.expand(len(h), -1, -1), h, h, key_padding_mask=pad)
        q = self.drop(self.qnorm(q))
        return [self.outs[j](q[:, j]) for j in range(3)]


def focal(logits, y):
    logp = F.log_softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** a.gamma * logp).mean()


def kd_loss(s, t):
    return sum(F.kl_div(F.log_softmax(s[j] / a.T, -1), F.softmax(t[j] / a.T, -1), reduction="batchmean")
               for j in range(3)) * a.T ** 2


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
def predict(m, D_, row, idx=None):
    m.eval()
    idx = torch.arange(len(D_["y"]), device=dev) if idx is None else idx
    outs = [[], [], []]
    for s in range(0, len(idx), 512):
        for j, o in enumerate(m(*tokens(D_, row, idx[s:s + 512]))):
            outs[j].append(o.float())
    return [torch.cat(o) for o in outs]


def train(row, seed, fi, ho, teacher=None):
    torch.manual_seed(seed)
    m = Probe().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * math.ceil(len(fi) / a.bs), pct_start=0.1)
    best = (-1.0, None, 0)
    for ep in range(a.epochs):
        m.train()
        perm = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            lg = m(*tokens(TR, row, b))
            loss = sum(focal(lg[j], TR["y"][b, j]) for j in range(3))
            if teacher is not None:
                with torch.no_grad():
                    teacher.eval(); t = teacher(*tokens(TR, "oracle", b))
                loss = loss + a.lam_kd * kd_loss(lg, t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        h = scores(predict(m, TR, row, ho), TR["y"][ho])
        sel = (h["verb"] + h["noun"] + h["action"]) / 3
        if sel > best[0]:
            best = (sel, copy.deepcopy(m.state_dict()), ep + 1)
        if a.every and ((ep + 1) % a.every == 0 or ep == 0):
            v = scores(predict(m, VA, row), VA["y"])
            print(f"  [{row} seed {seed}] epoch {ep + 1:3d}  held-out mean recall {sel:5.1f}{' *' if best[2] == ep + 1 else '  '}"
                  f"  val verb {v['verb']:5.1f}  object {v['noun']:5.1f}  action {v['action']:4.1f}", flush=True)
    m.load_state_dict(best[1])
    r = scores(predict(m, VA, row), VA["y"]); r["epoch"] = best[2]
    return r, m


rows = a.rows.split(",")
res = collections.defaultdict(list)
for seed in range(a.seeds):
    perm = torch.randperm(len(TR["y"]), generator=torch.Generator().manual_seed(seed)).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    teacher = None
    for row in sorted(rows, key=lambda r: r != "oracle"):
        if row.startswith("kd"):
            if teacher is None:
                _, teacher = train("oracle", seed, fi, ho)
            for q_ in teacher.parameters():
                q_.requires_grad_(False)
            r, _ = train(row, seed, fi, ho, teacher=teacher)
        else:
            r, m = train(row, seed, fi, ho)
            if row == "oracle":
                teacher = m
        res[row].append(r)
        print(f"{row:7s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  epoch {r['epoch']}", flush=True)

print(f"\n({a.seeds} seeds, mean +- sd; class-mean top-5 recall on validation_challenge)")
for row in rows:
    rs = res[row]
    ms = lambda k: (lambda t: f"{t.mean():5.1f} +- {t.std():3.1f}")(torch.tensor([r[k] for r in rs]))   # noqa: E731
    print(f"{row:7s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}")
for row, base in (("hist", "recent"), ("kd", "hist"), ("kd_rec", "recent")):
    if row in res and base in res:
        for k in ("verb", "noun", "action"):
            g = torch.tensor([x[k] - b[k] for x, b in zip(res[row], res[base])])
            print(f"gain {row} - {base} ({k}): {g.mean():+.1f} +- {g.std():.1f}")
json.dump({"args": vars(a), "results": res}, open(os.path.join(a.bank, f"teach{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)
