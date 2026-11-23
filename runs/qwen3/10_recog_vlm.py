"""Option 5, step 2: recognize each action segment from frozen VLM features (09_seg_feats.py), TSM, or both, and
compare with the TSM-only recognizer of 06_recog.py (validation top-1 action 35.7). Optionally (--export ROW) write a
recognizer file in the format of 06_recog.py (out-of-fold train predictions, validation predictions, probabilities,
the same histories and TSM segment features), so 04_probe.py can use the new predicted / soft history.
Inputs per segment: VLM = 4 frames x 9 grid cells (36 tokens, 4096-d); TSM = 1 token (2048-d, from 06_recog.py's
recog.pt "seg"). Model: 1 transformer block + mean pool -> action / verb / object; cross-entropy, label smoothing 0.1.
usage: python 10_recog_vlm.py /workspace/outputs/q3_seg /workspace/outputs/recog256v/recog.pt OUT_DIR
       [--rows tsm,vlm,vlm_tsm] [--export vlm_tsm]
"""
import argparse, csv, glob, json, math, os, re, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("seg"); ap.add_argument("recog"); ap.add_argument("out")
ap.add_argument("--rows", default="tsm,vlm,vlm_tsm")
ap.add_argument("--export", default="")
ap.add_argument("--csvs", default="/workspace/data/a101_anticip/data/CSVs", help="for the folds by recording (as 06_recog.py)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
t0 = time.time()


def load(split):
    parts = sorted(glob.glob(os.path.join(a.seg, f"{split}_*.pt")), key=lambda p: int(re.findall(r"_(\d+)\.pt$", p)[0]))
    ch = [torch.load(p) for p in parts]
    return torch.cat([c["frames"] for c in ch]).flatten(1, 2), torch.cat([c["y"] for c in ch]), torch.cat([c["inside"] for c in ch])


R = torch.load(a.recog)
V, Y, INS = {}, {}, {}
for s in ("train", "validation"):
    V[s], Y[s], INS[s] = load(s)
    assert torch.equal(Y[s], R[s]["y"]), f"{s}: segment features and recognizer file differ"
SIZES = R["sizes"]
mv = V["train"][:4000].float().reshape(-1, 4096); MV, SV = mv.mean(0), mv.std(0) + 1e-3
ts = R["train"]["seg"][:4000].float(); MT, ST = ts.mean(0), ts.std(0) + 1e-3
G = {s: {"vlm": ((V[s].float() - MV) / SV).half().to(dev), "tsm": ((R[s]["seg"].float() - MT) / ST).half().to(dev)[:, None],
         "y": Y[s].to(dev)} for s in V}
del V
print(f"train {len(G['train']['y'])} val {len(G['validation']['y'])} | loaded in {(time.time() - t0) / 60:.1f} min "
      f"| validation segments with 4+ frames inside: {(INS['validation'] >= 4).float().mean():.0%}", flush=True)
PARTS = {"tsm": ["tsm"], "vlm": ["vlm"], "vlm_tsm": ["vlm", "tsm"]}


class Rec(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.pv = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d)); self.pt = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
        self.kind = nn.Embedding(2, d); self.pos = nn.Embedding(64, d)
        self.block = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True,
                                                                      activation="gelu"), 1)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(0.2)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def forward(self, xs):
        hs = []
        for name, x in xs:
            h = (self.pv if name == "vlm" else self.pt)(x.float())
            hs.append(h + self.kind.weight[int(name == "tsm")] + self.pos(torch.arange(x.shape[1], device=x.device)))
        h = self.norm(self.block(torch.cat(hs, 1))).mean(1)
        return [o(self.drop(h)) for o in self.outs]


def inputs(s, row, idx):
    return [(p, G[s][p][idx]) for p in PARTS[row]]


def fit(row, idx, seed=0):
    torch.manual_seed(seed)
    m = Rec().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.epochs * math.ceil(len(idx) / 256), pct_start=0.1)
    for ep in range(a.epochs):
        m.train()
        for b in idx[torch.randperm(len(idx), device=dev)].split(256):
            lg = m(inputs("train", row, b))
            loss = sum(F.cross_entropy(lg[j].float(), G["train"]["y"][b, j], label_smoothing=0.1) for j in range(3))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


@torch.no_grad()
def predict(m, s, row, idx=None):
    m.eval()
    idx = torch.arange(len(G[s]["y"]), device=dev) if idx is None else idx
    out = [[], [], []]
    for b in idx.split(1024):
        for j, o in enumerate(m(inputs(s, row, b))):
            out[j].append(o.float())
    return [torch.cat(o) for o in out]


def acc(lg, y):
    return {k: round(100 * (lg[j].argmax(-1) == y[:, j]).float().mean().item(), 1) for j, k in enumerate(("action", "verb", "noun"))}


res = {}
ALL = torch.arange(len(G["train"]["y"]), device=dev)
for row in a.rows.split(","):
    m = fit(row, ALL)
    r = acc(predict(m, "validation", row), G["validation"]["y"]); res[row] = r
    print(f"{row:8s} validation top-1: action {r['action']:5.1f}  verb {r['verb']:5.1f}  object {r['noun']:5.1f}"
          f"   | {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"(TSM recognizer of 06_recog.py, 8 windows per segment: {R['acc']['validation']})")
json.dump(res, open(os.path.join(a.out, "recog_compare.json"), "w"), indent=1)

if a.export:
    row = a.export
    with open(os.path.join(a.csvs, "train.csv")) as f:                               # folds by recording, exactly as 06_recog.py
        seq_of = [r["video"].split("/")[0] for r in csv.DictReader(f) if r["video"].endswith("/C10119_rgb.mp4")]
    assert len(seq_of) == len(G["train"]["y"])
    fold_of = {q: i % a.folds for i, q in enumerate(sorted(set(seq_of)))}
    fold = torch.tensor([fold_of[q] for q in seq_of], device=dev)
    pred = torch.zeros_like(G["train"]["y"]); prob = torch.zeros(len(pred), sum(SIZES), dtype=torch.float16, device=dev)
    for k in range(a.folds):
        te = (fold == k).nonzero().flatten()
        m = fit(row, (fold != k).nonzero().flatten(), seed=k)
        lg = predict(m, "train", row, te)
        pred[te] = torch.stack([o.argmax(-1) for o in lg], 1); prob[te] = torch.cat([o.softmax(-1) for o in lg], -1).half()
        print(f"export fold {k + 1}/{a.folds}: top-1 {acc(lg, G['train']['y'][te])}", flush=True)
    m = fit(row, ALL, seed=99)
    lg = predict(m, "validation", row)
    out = {"acc": {"train_out_of_fold": acc([prob[:, :SIZES[0]].float(), prob[:, SIZES[0]:SIZES[0] + SIZES[1]].float(),
                                             prob[:, SIZES[0] + SIZES[1]:].float()], G["train"]["y"]),
                   "validation": acc(lg, G["validation"]["y"])}, "K": R["K"], "sizes": SIZES}
    out["train"] = {"y": R["train"]["y"], "pred": pred.cpu(), "prob": prob.cpu(), "hist": R["train"]["hist"], "seg": R["train"]["seg"]}
    out["validation"] = {"y": R["validation"]["y"], "pred": torch.stack([o.argmax(-1) for o in lg], 1).cpu(),
                         "prob": torch.cat([o.softmax(-1) for o in lg], -1).half().cpu(),
                         "hist": R["validation"]["hist"], "seg": R["validation"]["seg"]}
    torch.save(out, os.path.join(a.out, "recog.pt"))
    print(f"exported {row} recognizer: {out['acc']} -> {os.path.join(a.out, 'recog.pt')}", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min")
