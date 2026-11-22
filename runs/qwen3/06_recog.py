"""Option B, part 1: a past-action recognizer and the action history of every anticipation segment (camera C10119).
1. Segment features: for EVERY annotated segment (train.csv rows of this camera, validation_challenge.csv rows), the
   released TSM features between its start and end frame, max-pooled into 8 windows.
2. Recognizer: a small transformer over those 8 windows -> action / verb / object. Train predictions are
   OUT-OF-FOLD (5 folds by recording), so the history used in training is as noisy as at test time; validation
   predictions come from a recognizer trained on all of train.
   (Caveat: the released TSM backbone was itself fine-tuned on train, so out-of-fold train predictions can still be
   a bit better than validation ones.)
3. History: for each segment, the last K annotated segments of the same recording that END before the cut
   (cut = start - 30 frames), oldest first, right-aligned, -1 = none.
Saves <out>/recog.pt = {split: {"y": [n,3], "pred": [n,3], "prob": [n, 1064+17+90] softmax, "hist": [n,K]}, "acc"};
label order action, verb, object.
usage: python 06_recog.py /workspace/data/a101_feats/TSM_features/C10119_rgb /workspace/data/a101_anticip/data/CSVs
                          /workspace/outputs/recog [--K 8]
"""
import argparse, collections, csv, math, os, time
import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("lmdb"); ap.add_argument("csvs"); ap.add_argument("out")
ap.add_argument("--view", default="C10119_rgb")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"


def rows_of(split):
    out = []
    if split == "train":
        with open(os.path.join(a.csvs, "train.csv")) as f:
            for r in csv.DictReader(f):
                if r["video"].endswith(f"/{a.view}.mp4"):
                    out.append((r["video"].split("/")[0], int(r["start_frame"]), int(r["end_frame"]),
                                int(r["action_id"]), int(r["verb_id"]), int(r["noun_id"])))
    else:
        with open(os.path.join(a.csvs, "validation_challenge.csv")) as f:
            for r in csv.DictReader(f):
                out.append((r["video"].split("/")[0], int(r["start"]), int(r["end"]),
                            int(r["action"]), int(r["verb"]), int(r["noun"])))
    return out


segs = {s: rows_of(s) for s in ("train", "validation")}
by_seq = collections.defaultdict(list)
for s, rows in segs.items():
    for i, r in enumerate(rows):
        by_seq[r[0]].append((s, i))
SIZES = [max(r[j] for rows in segs.values() for r in rows) + 1 for j in (3, 4, 5)]
print({s: len(r) for s, r in segs.items()}, f"segments in {len(by_seq)} recordings | classes {SIZES}", flush=True)

# ------------------------------------------------------------------ 1. segment features
X = {s: np.zeros((len(r), 8, 2048), np.float16) for s, r in segs.items()}


def process(seq, frames):
    n = max(frames) + 1
    Fm = np.zeros((n, 2048), np.float32)
    for k, v in frames.items():
        Fm[k] = v
    for s, i in by_seq[seq]:
        _, st, en = segs[s][i][:3]
        edges = np.linspace(st, max(en, st + 1), 9).round().astype(int)
        for j in range(8):
            lo, hi = max(0, min(n - 1, edges[j])), max(0, min(n - 1, edges[j + 1]))
            X[s][i, j] = Fm[lo:max(lo, hi) + 1].max(0)


t0 = time.time()
env = lmdb.open(a.lmdb, readonly=True, lock=False, readahead=True)
cur, frames, done = None, {}, 0
with env.begin() as txn:
    for key, val in txn.cursor():
        k = key.decode(); seq = k.split("/")[0]
        if seq != cur:
            if cur in by_seq:
                process(cur, frames); done += 1
                if done % 50 == 0:
                    print(f"features: {done}/{len(by_seq)} recordings | {(time.time() - t0) / 60:.1f} min", flush=True)
            cur, frames = seq, {}
        if seq in by_seq:
            frames[int(k.rsplit("_", 1)[1].split(".")[0]) - 1] = np.frombuffer(val, np.float32)
    if cur in by_seq:
        process(cur, frames)
print(f"segment features done in {(time.time() - t0) / 60:.1f} min", flush=True)

Xg = {s: torch.from_numpy(X[s]).to(dev) for s in X}
Y = {s: torch.tensor([r[3:6] for r in segs[s]]).to(dev) for s in segs}
mu = Xg["train"][:4000].float().mean((0, 1)); sd = Xg["train"][:4000].float().std((0, 1)) + 1e-3


# ------------------------------------------------------------------ 2. recognizer
class Rec(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d)); self.pos = nn.Parameter(torch.zeros(1, 8, d))
        self.block = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True,
                                                                      activation="gelu"), 1)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(0.2)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def forward(self, x):
        h = self.norm(self.block(self.inp((x.float() - mu) / sd) + self.pos)).mean(1)
        return [o(self.drop(h)) for o in self.outs]


def fit(idx, seed=0):
    torch.manual_seed(seed)
    m = Rec().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(idx) / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        m.train()
        perm = idx[torch.randperm(len(idx), device=dev)]
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            lg = m(Xg["train"][b])
            loss = sum(F.cross_entropy(lg[j].float(), Y["train"][b, j], label_smoothing=0.1) for j in range(3))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


@torch.no_grad()
def predict(m, x):
    m.eval()
    out = [[], [], []]
    for s in range(0, len(x), 1024):
        for j, o in enumerate(m(x[s:s + 1024])):
            out[j].append(o.argmax(-1))
    return torch.stack([torch.cat(o) for o in out], 1)


@torch.no_grad()
def probs(m, x):
    m.eval()
    out = []
    for s in range(0, len(x), 1024):
        out.append(torch.cat([o.float().softmax(-1) for o in m(x[s:s + 1024])], -1).half())
    return torch.cat(out)


def acc(p, y):
    return {k: round(100 * (p[:, j] == y[:, j]).float().mean().item(), 1) for j, k in enumerate(("action", "verb", "noun"))}


seqs = sorted({r[0] for r in segs["train"]})
fold_of = {q: i % a.folds for i, q in enumerate(seqs)}
fold = torch.tensor([fold_of[r[0]] for r in segs["train"]], device=dev)
pred_tr = torch.zeros_like(Y["train"])
prob_tr = torch.zeros(len(Y["train"]), sum(SIZES), dtype=torch.float16, device=dev)
for k in range(a.folds):
    m = fit((fold != k).nonzero().flatten(), seed=k)
    te = (fold == k).nonzero().flatten()
    pred_tr[te] = predict(m, Xg["train"][te]); prob_tr[te] = probs(m, Xg["train"][te])
    print(f"fold {k + 1}/{a.folds}: top-1 {acc(pred_tr[te], Y['train'][te])} | {(time.time() - t0) / 60:.1f} min", flush=True)
m = fit(torch.arange(len(Y["train"]), device=dev), seed=99)
pred_va = predict(m, Xg["validation"]); prob_va = probs(m, Xg["validation"])
ACC = {"train_out_of_fold": acc(pred_tr, Y["train"]), "validation": acc(pred_va, Y["validation"])}
print(f"recognizer top-1 accuracy: {ACC}", flush=True)


# ------------------------------------------------------------------ 3. action history per segment
def histories(rows):
    H = torch.full((len(rows), a.K), -1, dtype=torch.long)
    per = collections.defaultdict(list)
    for i, r in enumerate(rows):
        per[r[0]].append(i)
    for seq, ids in per.items():
        ends = sorted(ids, key=lambda i: rows[i][2])
        for i in ids:
            cut = rows[i][1] - 30
            past = [j for j in ends if rows[j][2] <= cut][-a.K:]
            if past:
                H[i, a.K - len(past):] = torch.tensor(past)
    return H


res = {"acc": ACC, "K": a.K, "sizes": SIZES}
for s, p, pr in (("train", pred_tr, prob_tr), ("validation", pred_va, prob_va)):
    H = histories(segs[s])
    res[s] = {"y": Y[s].cpu(), "pred": p.cpu(), "prob": pr.cpu(), "hist": H}
    n = (H >= 0).sum(1).float()
    print(f"{s}: past actions per segment mean {n.mean():.1f}, median {n.median():.0f}, max {n.max():.0f}, "
          f"at the cap ({a.K}) {(n == a.K).float().mean():.1%}, none {(n == 0).float().mean():.1%}", flush=True)
torch.save(res, os.path.join(a.out, "recog.pt"))
print(f"saved {os.path.join(a.out, 'recog.pt')} | done in {(time.time() - t0) / 60:.1f} min")
