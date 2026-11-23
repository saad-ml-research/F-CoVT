"""VLM ladder, rungs 1-4, on the official anticipation set (camera C10119): one attentive probe (same as the TSM
runs: tokens -> 2 transformer blocks -> 3 queries -> verb / object / action, focal loss) over any mix of
  Qwen3-VL frame tokens (03_features.py): 8 history frames (last 60 s) + 4 recent frames (last 2 s), at a chosen layer
  Qwen3-VL last prompt token
  TSM tokens (runs/tsm/01_bank.py): 30 history windows + 8 recent windows (+ 8 future windows, teacher only)
Rows (append @18 / @27 / @36 to pick the Qwen3-VL layer; default 27; with --recog, append +hT / +hTP / +hP to add the
past-action history as label tokens: true labels in train and val / true in train, predicted in val / predicted in both;
row hlab = history labels only):
  q_last    rung 1: the VLM's last prompt token only
  q_rec     rung 1: the 4 recent VLM frames
  q_hist    rung 2a: 8 history + 4 recent VLM frames
  tsm_hist  reference: TSM history + recent (our best student without teaching)
  fuse      rung 3: VLM frames + TSM history + recent
  oracle    TSM history + recent + REAL future (cheating ceiling; the teacher)
  fuse_kd   rung 4: fuse student + KD from the oracle teacher (weight --lam_kd, temperature --T)
  tsm_kd    reference: TSM student + the same teacher
The VLM and TSM banks list the same segments in the same order (checked on the labels).
usage: python 04_probe.py /workspace/outputs/q3_feats /workspace/outputs/tsm_bank
       [--rows q_last,q_rec,q_hist@18,q_hist@27,q_hist@36,fuse,fuse_kd,tsm_hist,tsm_kd] [--seeds 3]
"""
import argparse, collections, copy, glob, json, math, os, re, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("q3"); ap.add_argument("tsm")
ap.add_argument("--rows", default="q_last,q_rec,q_hist@18,q_hist@27,q_hist@36,tsm_hist,fuse,fuse_kd,tsm_kd")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--lam_kd", type=float, default=2.0)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--tag", default="")
ap.add_argument("--fuse", default="", help='late fusion of val predictions: "rowA|rowB,..." (sum of log-probs)')
ap.add_argument("--recog", default="", help="recog.pt from 06_recog.py: enables history rows (+hT, +hTP, +hP)")
a = ap.parse_args()
dev = "cuda"


def load_q3(split):
    parts = sorted(glob.glob(os.path.join(a.q3, f"{split}_*.pt")), key=lambda p: int(re.findall(r"_(\d+)\.pt$", p)[0]))
    ch = [torch.load(p) for p in parts]
    return {"frames": torch.cat([c["frames"] for c in ch]), "mask": torch.cat([c["mask"] for c in ch]),
            "last": torch.cat([c["last"] for c in ch]), "y": torch.cat([c["y"] for c in ch]), "layers": ch[0]["layers"]}


Q = {s: load_q3(s) for s in ("train", "validation")}
T = {s: torch.load(os.path.join(a.tsm, f"{s}.pt")) for s in ("train", "validation")}
for s in Q:
    assert len(Q[s]["y"]) == len(T[s]["y"]) and torch.equal(Q[s]["y"], T[s]["y"]), f"{s}: VLM and TSM banks differ"
LAYERS = Q["train"]["layers"]
SIZES = [int(max(Q["train"]["y"][:, j].max(), Q["validation"]["y"][:, j].max())) + 1 for j in range(3)]
print(f"train {len(Q['train']['y'])}  val {len(Q['validation']['y'])} | VLM layers {LAYERS} | classes {SIZES}", flush=True)

# token groups: name -> (source, kind id, n tokens); positions = index inside the group
GROUPS = {"q_last": 0, "q_hist": 1, "q_rec": 2, "t_hist": 3, "t_rec": 4, "t_fut": 5, "h": 6}
ROWS = {"q_last": ["q_last"], "q_rec": ["q_rec"], "q_hist": ["q_hist", "q_rec"], "tsm_hist": ["t_hist", "t_rec"],
        "tsm_kd": ["t_hist", "t_rec"], "fuse": ["q_hist", "q_rec", "t_hist", "t_rec"],
        "fuse_kd": ["q_hist", "q_rec", "t_hist", "t_rec"], "oracle": ["t_hist", "t_rec", "t_fut"], "hlab": []}
ROWS["fuse_kdH"], ROWS["tsm_kdH"] = ROWS["fuse_kd"], ROWS["tsm_kd"]     # same students, taught by oracleH
# teachers: oracle = TSM past + real future; oracleH = the same + the TRUE past-action labels (privileged)
TEACH = {"oracle": ("oracle", ROWS["oracle"]), "oracleH": ("oracle+hT8", ROWS["oracle"] + ["h"])}   # oracleH: last 8 true labels
# history modes (row suffix): labels of the past actions used in (training, validation)
HMODE = {"hT": ("true", "true"), "hTP": ("true", "pred"), "hP": ("pred", "pred"), "hS": ("soft", "soft"),
         "hV": ("seg", "seg")}       # hV = aligned video history: TSM features of each past action segment
NOISE = sorted({int(x) for x in re.findall(r"hN(\d+)", a.rows)})   # hN30 = true labels with 30% replaced at random


def hmode(hm):
    """'hS' / 'hS64' / 'hTP' ... -> (train source, val source) as 'src:length'; length = last N past actions (default 8)."""
    if re.fullmatch(r"hN\d+", hm):
        return (f"n{hm[2:]}:8",) * 2
    m = re.fullmatch(r"(hTP|hT|hP|hS|hV)(\d*)", hm)
    L = int(m.group(2) or 8)
    return tuple(f"{src}:{L}" for src in HMODE[m.group(1)])
PSUM = sum(SIZES)                    # soft history token = recognizer probabilities (action | verb | object)
R = torch.load(a.recog) if a.recog else None
if R:
    print(f"recognizer top-1 accuracy: {R['acc']}", flush=True)
    for s_ in ("train", "validation"):
        assert torch.equal(R[s_]["y"], Q[s_]["y"]), f"{s_}: recog and VLM banks differ"


def stats(x):
    x = x[:4000].float().reshape(-1, x.shape[-1])
    return x.mean(0), x.std(0) + 1e-3


cache = {}


def gpu_data(layer):
    """standardized tokens on the GPU for one VLM layer (TSM shared)."""
    if layer in cache:
        return cache[layer]
    li = LAYERS.index(layer)
    out = {}
    pick = lambda fr: fr[:, :, :, li] if fr.dim() == 5 else fr[:, :, li][:, :, None]   # [n, 12, cells, D]  # noqa: E731
    mq, sq = stats(pick(Q["train"]["frames"])[:, 8:]); ml, sl = stats(Q["train"]["last"][:, li:li + 1])
    mt, st = stats(T["train"]["recent"])
    for s in ("train", "validation"):
        fr = ((pick(Q[s]["frames"]).float() - mq) / sq).half()
        n, _, c, d = fr.shape                                            # c = grid cells per frame (1 = mean)
        out[s] = {"q_hist": fr[:, :8].reshape(n, 8 * c, d).to(dev), "q_rec": fr[:, 8:].reshape(n, 4 * c, d).to(dev),
                  "q_hist_m": Q[s]["mask"][:, :8].repeat_interleave(c, 1).to(dev),
                  "q_last": ((Q[s]["last"][:, li:li + 1].float() - ml) / sl).half().to(dev),
                  "t_hist": ((T[s]["hist"].float() - mt) / st).half().to(dev), "t_hist_m": T[s]["hist_mask"].to(dev),
                  "t_rec": ((T[s]["recent"].float() - mt) / st).half().to(dev),
                  "t_fut": ((T[s]["fut"].float() - mt) / st).half().to(dev), "y": Q[s]["y"].to(dev)}
        if R:
            H = R[s]["hist"]; ok = H >= 0
            pad = torch.tensor(SIZES)
            for src, lab in (("true", R[s]["y"]), ("pred", R[s]["pred"])):
                v = lab[H.clamp(min=0)]                                     # [n, K, 3]
                out[s]["h_" + src] = torch.where(ok[..., None], v, pad).to(dev)
            out[s]["h_idx"] = H.to(dev)
            if "seg" in R[s]:
                out[s]["seg"] = ((R[s]["seg"].float() - mt) / st).half().to(dev)
            g = torch.Generator().manual_seed(123 + (s == "validation"))
            for p in NOISE:      # replace p% of the history slots by a random TRAIN segment's (action, verb, object)
                v = R[s]["y"][H.clamp(min=0)]
                rnd = R["train"]["y"][torch.randint(len(R["train"]["y"]), H.shape, generator=g)]
                flip = torch.rand(H.shape, generator=g) < p / 100
                v = torch.where(flip[..., None], rnd, v)
                out[s][f"h_n{p}"] = torch.where(ok[..., None], v, pad).to(dev)
            if "prob" in R[s]:
                out[s]["prob"] = R[s]["prob"].to(dev)                          # [n, PSUM], gathered per batch
            m_ = ok.clone(); m_[:, -1] = True                               # last slot always on ("no history" token)
            out[s]["h_m"] = m_.to(dev)
    cache.clear(); cache[layer] = out
    return out


def tokens(D, groups, idx, hsrc="true:8"):
    xs, ms, kinds, posi = [], [], [], []
    for g in groups:
        if g == "h":
            src, L = hsrc.split(":"); L = int(L)
            if src in ("soft", "seg"):
                H = D["h_idx"][idx][:, -L:]
                x = D["prob" if src == "soft" else "seg"][H.clamp(min=0)] * (H >= 0)[..., None]
            else:
                x = D["h_" + src][idx][:, -L:]
            m = D["h_m"][idx][:, -L:]
        else:
            x = D[g][idx]
            m = D[g + "_m"][idx] if g + "_m" in D else torch.ones(len(idx), x.shape[1], dtype=torch.bool, device=dev)
        n = x.shape[1]
        xs.append(x)
        ms.append(m)
        kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(n, device=dev))
    return xs, torch.cat(ms, 1), torch.cat(kinds), torch.cat(posi)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        d = a.d
        self.inq = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d))
        self.int = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
        self.hemb = nn.ModuleList(nn.Embedding(n + 1, d) for n in SIZES)          # past action / verb / object labels
        self.kind = nn.Embedding(len(GROUPS), d); self.pos = nn.Embedding(512, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, a.drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, a.blocks)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, dropout=a.drop, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        self.drop = nn.Dropout(a.drop)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def embed(self, x):
        if x.dtype == torch.long:                                       # hard labels
            return sum(e(x[..., j]) for j, e in enumerate(self.hemb))
        if x.shape[-1] == PSUM:                                         # soft labels: expected label embedding
            parts = x.float().split(SIZES, -1)
            return sum(p @ e.weight[:n] for p, e, n in zip(parts, self.hemb, SIZES))
        return (self.inq if x.shape[-1] == 4096 else self.int)(x.float())

    def forward(self, xs, mask, kinds, posi):
        h = torch.cat([self.embed(x) for x in xs], 1)
        h = h + self.kind(kinds) + self.pos(posi)
        pad = ~mask
        h = self.norm(self.blocks(h, src_key_padding_mask=pad))
        q, _ = self.xattn(self.q.expand(len(h), -1, -1), h, h, key_padding_mask=pad)
        q = self.drop(self.qnorm(q))
        return [self.outs[j](q[:, j]) for j in range(3)]


def focal(lg, y):
    lp = F.log_softmax(lg.float(), -1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - lp.exp()) ** a.gamma * lp).mean()


def kd_loss(s, t):
    return sum(F.kl_div(F.log_softmax(s[j] / a.T, -1), F.softmax(t[j] / a.T, -1), reduction="batchmean")
               for j in range(3)) * a.T ** 2


def cm_recall(lg, y):
    hit = (lg.topk(5, -1).indices == y[:, None]).any(-1)
    per = collections.defaultdict(list)
    for c, h in zip(y.tolist(), hit.tolist()):
        per[c].append(h)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def scores(lg, y):
    """class-mean top-5 recall (official metric) + plain top-1 / top-5 accuracy, for action / verb / noun."""
    r = {"action": cm_recall(lg[0], y[:, 0]), "verb": cm_recall(lg[1], y[:, 1]), "noun": cm_recall(lg[2], y[:, 2])}
    for j, k in enumerate(("action", "verb", "noun")):
        top = lg[j].topk(5, -1).indices
        r[f"{k}_top1"] = 100 * (top[:, 0] == y[:, j]).float().mean().item()
        r[f"{k}_top5"] = 100 * (top == y[:, j, None]).any(-1).float().mean().item()
    return r


@torch.no_grad()
def predict(m, D, groups, idx=None, hsrc="true:8"):
    m.eval()
    idx = torch.arange(len(D["y"]), device=dev) if idx is None else idx
    outs = [[], [], []]
    for s in range(0, len(idx), 512):
        for j, o in enumerate(m(*tokens(D, groups, idx[s:s + 512], hsrc))):
            outs[j].append(o.float())
    return [torch.cat(o) for o in outs]


def train(name, base, D, seed, fi, ho, teacher=None):
    torch.manual_seed(seed)
    base, _, hm = base.partition("+")
    groups = ROWS[base] + (["h"] if hm else [])
    htr, hte = hmode(hm) if hm else ("true:8", "true:8")
    m = Probe().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * math.ceil(len(fi) / a.bs), pct_start=0.1)
    best = (-1.0, None, 0)
    for ep in range(a.epochs):
        m.train()
        perm = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            lg = m(*tokens(D["train"], groups, b, htr))
            loss = sum(focal(lg[j], D["train"]["y"][b, j]) for j in range(3))
            if teacher is not None:
                with torch.no_grad():
                    teacher[0].eval(); t = teacher[0](*tokens(D["train"], teacher[1], b, teacher[2]))
                loss = loss + a.lam_kd * kd_loss(lg, t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        h = scores(predict(m, D["train"], groups, ho, htr), D["train"]["y"][ho])
        sel = (h["verb"] + h["noun"] + h["action"]) / 3
        if sel > best[0]:
            best = (sel, copy.deepcopy(m.state_dict()), ep + 1)
        if a.every and ((ep + 1) % a.every == 0 or ep == 0):
            v = scores(predict(m, D["validation"], groups, hsrc=hte), D["validation"]["y"])
            print(f"  [{name} seed {seed}] epoch {ep + 1:3d}  held-out mean recall {sel:5.1f}{' *' if best[2] == ep + 1 else '  '}"
                  f"  val verb {v['verb']:5.1f}  object {v['noun']:5.1f}  action {v['action']:4.1f}", flush=True)
    m.load_state_dict(best[1])
    lg = predict(m, D["validation"], groups, hsrc=hte)
    VAL[name] = [F.log_softmax(o, -1).half() for o in lg]                 # kept for late fusion
    r = scores(lg, D["validation"]["y"]); r["epoch"] = best[2]
    return r, m


rows = a.rows.split(",")
FUSE = [f.split("|") for f in a.fuse.split(",") if f]
res = collections.defaultdict(list)
VAL, TIMES, t_start = {}, {}, time.time()
JOBS = rows + [t for t in ("oracle", "oracleH") if any(r_.split("@")[0].split("+")[0].endswith("_kdH" if t == "oracleH" else "_kd") for r_ in rows)]


def eta(seed, done_names):
    """minutes left: known per-row times (from seed 0), mean time for rows not seen yet."""
    known = list(TIMES.values()); avg = sum(known) / len(known) if known else 0
    left = [n for n in JOBS if n not in done_names] + JOBS * (a.seeds - seed - 1)
    m = sum(TIMES.get(n, avg) for n in left) / 60
    fin = time.strftime("%H:%M UTC", time.gmtime(time.time() + m * 60))
    return f"[timer] {(time.time() - t_start) / 60:.0f} min elapsed, ~{m:.0f} min left (done around {fin})"


for seed in range(a.seeds):
    done_names = []
    perm = torch.randperm(len(Q["train"]["y"]), generator=torch.Generator().manual_seed(seed)).to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    teachers = {}
    for name in rows:
        base, layer = (name.split("@") + ["27"])[:2]
        D = gpu_data(int(layer))
        b0 = base.split("+")[0]
        if b0.endswith("_kd") or b0.endswith("_kdH"):
            tk = "oracleH" if b0.endswith("_kdH") else "oracle"
            if tk not in teachers:
                t_ = time.time()
                r0, tm = train(tk, TEACH[tk][0], D, seed, fi, ho)
                TIMES.setdefault(tk, time.time() - t_); done_names.append(tk)
                res[tk].append(r0)
                print(f"{tk:12s} seed {seed}: verb {r0['verb']:5.1f}  object {r0['noun']:5.1f}  action {r0['action']:4.1f}", flush=True)
                for p in tm.parameters():
                    p.requires_grad_(False)
                teachers[tk] = (tm, TEACH[tk][1], "true:8")
            t_ = time.time()
            r, _ = train(name, base, D, seed, fi, ho, teacher=teachers[tk])
        else:
            t_ = time.time()
            r, _ = train(name, base, D, seed, fi, ho)
        TIMES.setdefault(name, time.time() - t_); done_names.append(name)
        res[name].append(r)
        print(f"{name:12s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  epoch {r['epoch']}", flush=True)
        print(eta(seed, done_names), flush=True)
    for A, B in FUSE:
        fl = [VAL[A][j].float() + VAL[B][j].float() for j in range(3)]
        r = scores(fl, D["validation"]["y"]); res[f"{A}|{B}"].append(r)
        print(f"{A + '|' + B:24s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  (late fusion)", flush=True)

print(f"\n({a.seeds} seeds, mean +- sd; class-mean top-5 recall on validation_challenge, camera C10119)")
for name in [t for t in TEACH if t in res] + rows + [f"{A}|{B}" for A, B in FUSE]:
    rs = res[name]
    ms = lambda k: (lambda t: f"{t.mean():5.1f} +- {t.std():3.1f}")(torch.tensor([r[k] for r in rs]))   # noqa: E731
    print(f"{name:12s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}"
          f" | action top-5 acc {ms('action_top5')}")
json.dump({"args": vars(a), "results": res}, open(os.path.join(a.q3, f"probe{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)
