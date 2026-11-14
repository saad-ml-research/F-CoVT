"""Model A v2, training the imagination: learned future prediction on the V-JEPA 2 token bank (01_bank.py).
Future slots (--slots): which of the 8 future time slots (0.25 s each, 8 = right after the cut) are imagined/used:
  8-10 = the 0.75 s BEFORE the next action starts, 11-15 = while it happens, 8-15 = all.
Rows (all with the same attentive probe + focal loss + recipe as 02_probe.py; 3 seeds):
  enc       past tokens only
  vjepa     + the frozen pretrained V-JEPA 2 predictor's tokens for the slots
  oracle    + the REAL future tokens for the slots (cheating ceiling; also the teacher for "distill")
  jepa      our predictor, trained with the JEPA loss only (L1 to the real future tokens, epoch by held-out JEPA),
            starts from V-JEPA 2's guess and learns a correction; then frozen, probe reads past + its guess
  distill   predictor + probe trained together: focal(labels) + lam_j * JEPA + lam_kd * KL(student || oracle teacher)
            -> imagine the parts of the future that change the decision (learning with privileged information)
Reported on validation: verb / object / action class-mean top-5 recall, action top-1, and for learned guesses
"explained" = 1 - |guess - real|^2 / |V-JEPA guess - real|^2 (> 0: better than the pretrained predictor).
usage: python 03_think.py /workspace/outputs/vjepa_bank --rows oracle --slots 8-10        (before/during split)
       python 03_think.py /workspace/outputs/vjepa_bank --rows enc,vjepa,jepa,distill,oracle --slots 8-15
"""
import argparse, collections, copy, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("bank")
ap.add_argument("--rows", default="enc,vjepa,jepa,distill,oracle")
ap.add_argument("--slots", default="8-15")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--pred_layers", type=int, default=4)
ap.add_argument("--pred_epochs", type=int, default=40)
ap.add_argument("--pred_lr", type=float, default=3e-4)
ap.add_argument("--lam_j", type=float, default=1.0)
ap.add_argument("--lam_kd", type=float, default=1.0)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"
SIZES = [1400, 32, 100]

tr, va = (torch.load(os.path.join(a.bank, f"{s}.pt")) for s in ("train", "validation"))
FS0 = tr["first_future_slot"]
NPS, NSP, D = tr["past"].shape[1], tr["past"].shape[2], tr["past"].shape[-1]
lo, hi = (int(x) for x in a.slots.split("-"))
SL = list(range(lo, hi + 1))
NS = len(SL)
sel = torch.tensor([s - FS0 for s in SL])
mu = tr["past"][:2000].float().mean((0, 1, 2)); sd = tr["past"][:2000].float().std((0, 1, 2)) + 1e-3


def prep(d):
    return {"past": d["past"].to(dev), "vj": d["pred"][:, sel].to(dev), "real": d["real"][:, sel].to(dev),
            "y": d["y"].to(dev)}


TR, VA = prep(tr), prep(va)
MU, SD = mu.to(dev), sd.to(dev)
norm = lambda x: (x.float() - MU) / SD                                   # noqa: E731
denorm = lambda x: x * SD + MU                                           # noqa: E731
print(f"train {len(TR['y'])}  val {len(VA['y'])} | future slots {SL} = {(lo - 11) * 0.25:+.2f} .. {(hi - 10) * 0.25:+.2f} s"
      f" relative to the next action's start | rows {a.rows}", flush=True)


# ------------------------------------------------------------------ probe (as 02_probe.py)
class Probe(nn.Module):
    def __init__(self, n_fut):
        super().__init__()
        d = a.d
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.time = nn.Embedding(16, d); self.kind = nn.Embedding(2, d); self.space = nn.Embedding(NSP, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, a.blocks)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, dropout=a.drop, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        self.drop = nn.Dropout(a.drop)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)
        tim = torch.cat([torch.arange(NPS), torch.tensor(SL[:n_fut], dtype=torch.long)]).repeat_interleave(NSP)
        kind = torch.cat([torch.zeros(NPS), torch.ones(n_fut)]).long().repeat_interleave(NSP)
        spc = torch.arange(NSP).repeat(NPS + n_fut)
        self.register_buffer("tim", tim, persistent=False)
        self.register_buffer("kd", kind, persistent=False)
        self.register_buffer("spc", spc, persistent=False)

    def forward(self, past, fut=None):                                  # raw-scale tokens [B, S, 16, D]
        x = past if fut is None else torch.cat([past, fut], 1)
        B, S = x.shape[:2]
        x = self.inp(x.reshape(B, S * NSP, D).float())
        L = x.shape[1]
        x = x + self.time(self.tim[:L]) + self.kind(self.kd[:L]) + self.space(self.spc[:L])
        x = self.norm(self.blocks(x))
        q, _ = self.xattn(self.q.expand(B, -1, -1), x, x)
        q = self.drop(self.qnorm(q))
        return [self.outs[j](q[:, j]) for j in range(3)]


# ------------------------------------------------------------------ our predictor
class Predictor(nn.Module):
    """reads past tokens + V-JEPA 2's guess for the slots; outputs a correction: guess = V-JEPA guess + delta."""
    def __init__(self):
        super().__init__()
        d = a.d
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.time = nn.Embedding(16, d); self.kind = nn.Embedding(2, d); self.space = nn.Embedding(NSP, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, a.pred_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # starts as V-JEPA 2's guess
        self.register_buffer("tim", torch.cat([torch.arange(NPS), torch.tensor(SL, dtype=torch.long)]).repeat_interleave(NSP), persistent=False)
        self.register_buffer("kd", torch.cat([torch.zeros(NPS), torch.ones(NS)]).long().repeat_interleave(NSP), persistent=False)
        self.register_buffer("spc", torch.arange(NSP).repeat(NPS + NS), persistent=False)

    def forward(self, past, vj):                                        # standardized [B, *, 16, D] -> [B, NS, 16, D]
        B = past.shape[0]
        x = torch.cat([past, vj], 1).reshape(B, -1, D)
        x = self.inp(x) + self.time(self.tim) + self.kind(self.kd) + self.space(self.spc)
        h = self.norm(self.enc(x))[:, NPS * NSP:]
        return vj + self.out(h).view(B, NS, NSP, D)


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


def fut_of(row, D_, idx, pred=None):
    """future tokens (raw scale) a row's probe reads, or None."""
    if row == "enc":
        return None
    if row == "vjepa":
        return D_["vj"][idx]
    if row == "oracle":
        return D_["real"][idx]
    return denorm(pred(norm(D_["past"][idx]), norm(D_["vj"][idx])))


@torch.no_grad()
def predict(m, D_, row, pred=None, idx=None):
    m.eval()
    if pred is not None:
        pred.eval()
    idx = torch.arange(len(D_["y"]), device=dev) if idx is None else idx
    outs = [[], [], []]
    for s in range(0, len(idx), 256):
        b = idx[s:s + 256]
        for j, o in enumerate(m(D_["past"][b], fut_of(row, D_, b, pred))):
            outs[j].append(o.float())
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def explained(pred, D_):
    pred.eval()
    num = den = 0.0
    for s in range(0, len(D_["y"]), 256):
        b = slice(s, s + 256)
        g = pred(norm(D_["past"][b]), norm(D_["vj"][b]))
        r, v = norm(D_["real"][b]), norm(D_["vj"][b])
        num += ((g - r) ** 2).sum().item(); den += ((v - r) ** 2).sum().item()
    return 1 - num / den


def train_jepa(seed, fi, ho):
    torch.manual_seed(seed)
    p = Predictor().to(dev)
    opt = torch.optim.AdamW(p.parameters(), lr=a.pred_lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.pred_lr, total_steps=a.pred_epochs * math.ceil(len(fi) / a.bs),
                                                pct_start=0.05)
    best = (math.inf, None, 0)
    for ep in range(a.pred_epochs):
        p.train()
        perm = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            loss = F.l1_loss(p(norm(TR["past"][b]), norm(TR["vj"][b])), norm(TR["real"][b]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        p.eval()
        with torch.no_grad():
            hl = sum(F.l1_loss(p(norm(TR["past"][b]), norm(TR["vj"][b])), norm(TR["real"][b]), reduction="sum").item()
                     for b in ho.split(256))
        if hl < best[0]:
            best = (hl, copy.deepcopy(p.state_dict()), ep + 1)
        if a.every and ((ep + 1) % a.every == 0 or ep == 0):
            print(f"    [jepa predictor seed {seed}] epoch {ep + 1:3d}  held-out L1 {hl / (len(ho) * NS * NSP * D):.4f}"
                  f"{' *' if best[2] == ep + 1 else '  '}  val explained {explained(p, VA):+.3f}", flush=True)
            p.train()
    p.load_state_dict(best[1]); p.eval()
    for q_ in p.parameters():
        q_.requires_grad_(False)
    return p


def train_probe(row, seed, fi, ho, pred=None, teacher=None):
    torch.manual_seed(seed)
    m = Probe(0 if row == "enc" else NS).to(dev)
    params = list(m.parameters()) + (list(pred.parameters()) if row == "distill" else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * math.ceil(len(fi) / a.bs), pct_start=0.1)
    best = (-1.0, None, None, 0)
    for ep in range(a.epochs):
        m.train()
        if row == "distill":
            pred.train()
        perm = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            if row == "distill":
                g = pred(norm(TR["past"][b]), norm(TR["vj"][b]))
                lg = m(TR["past"][b], denorm(g))
            else:
                lg = m(TR["past"][b], fut_of(row, TR, b, pred))
            loss = sum(focal(lg[j], TR["y"][b, j]) for j in range(3))
            if row == "distill":
                loss = loss + a.lam_j * F.l1_loss(g, norm(TR["real"][b]))
                with torch.no_grad():
                    teacher.eval(); t = teacher(TR["past"][b], TR["real"][b])
                loss = loss + a.lam_kd * kd_loss(lg, t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        h = scores(predict(m, TR, row, pred, ho), TR["y"][ho])
        selv = (h["verb"] + h["noun"] + h["action"]) / 3
        if selv > best[0]:
            best = (selv, copy.deepcopy(m.state_dict()), copy.deepcopy(pred.state_dict()) if row == "distill" else None, ep + 1)
        if a.every and ((ep + 1) % a.every == 0 or ep == 0):
            v = scores(predict(m, VA, row, pred), VA["y"])
            ex = f"  explained {explained(pred, VA):+.3f}" if row == "distill" else ""
            print(f"  [{row} seed {seed}] epoch {ep + 1:3d}  held-out mean recall {selv:5.1f}{' *' if best[3] == ep + 1 else '  '}"
                  f"  val verb {v['verb']:5.1f}  object {v['noun']:5.1f}  action {v['action']:4.1f}{ex}", flush=True)
    m.load_state_dict(best[1])
    if row == "distill":
        pred.load_state_dict(best[2])
    r = scores(predict(m, VA, row, pred), VA["y"]); r["epoch"] = best[3]
    if pred is not None:
        r["explained"] = explained(pred, VA)
    return r, m


rows = a.rows.split(",")
res = collections.defaultdict(list)
for seed in range(a.seeds):
    perm = torch.randperm(len(TR["y"]), generator=torch.Generator().manual_seed(seed)).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    teacher = None
    for row in sorted(rows, key=lambda r: r != "oracle"):              # oracle first: it is the distill teacher
        if row == "jepa":
            r, _ = train_probe(row, seed, fi, ho, pred=train_jepa(seed, fi, ho))
        elif row == "distill":
            if teacher is None:
                _, teacher = train_probe("oracle", seed, fi, ho)
            for q_ in teacher.parameters():
                q_.requires_grad_(False)
            r, _ = train_probe(row, seed, fi, ho, pred=Predictor().to(dev), teacher=teacher)
        else:
            r, m = train_probe(row, seed, fi, ho)
            if row == "oracle":
                teacher = m
        res[row].append(r)
        ex = f"  explained {r['explained']:+.3f}" if "explained" in r else ""
        print(f"{row:8s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  epoch {r['epoch']}{ex}", flush=True)

print(f"\nslots {SL}  ({a.seeds} seeds, mean +- sd)")
for row in rows:
    rs = res[row]
    ms = lambda k: (lambda t: f"{t.mean():5.1f} +- {t.std():3.1f}")(torch.tensor([r[k] for r in rs]))   # noqa: E731
    ex = f" | explained {torch.tensor([r['explained'] for r in rs]).mean():+.3f}" if "explained" in rs[0] else ""
    print(f"{row:8s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}{ex}")
for row in ("vjepa", "jepa", "distill"):
    if "enc" in res and row in res:
        g = torch.tensor([x["verb"] - e["verb"] for x, e in zip(res[row], res["enc"])])
        print(f"verb gain {row} - enc: {g.mean():+.1f} +- {g.std():.1f}")
json.dump({"args": vars(a), "results": res},
          open(os.path.join(a.bank, f"think_{a.slots}{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)
