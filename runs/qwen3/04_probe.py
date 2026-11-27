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
ap.add_argument("--texts", default="", help='style=path,... from 12_text_styles.py: rows +hXP-<style> (steps), +hXL-<style> (final state)')
ap.add_argument("--lam_hand", type=float, default=0.5, help="weight of the hand-forecast loss for *_kdPA rows")
ap.add_argument("--hands", default="", help="hand_bank.pt from 15_hand_bank.py: rows *_kdP (student + hands), *_kdF (teacher + hands), *_kdPF (both)")
ap.add_argument("--thoughts", default="", help="dir from 13_reason.py thoughts: rows fuse_th_kd (+ 2 reasoning tokens)")
ap.add_argument("--train_subset", default="", help="json list of train indices: train / held-out only on these")
ap.add_argument("--rl_epochs", type=int, default=3, help="GRPO fine-tuning epochs for rows with #<reward>")
ap.add_argument("--rl_lr", type=float, default=1e-5)
ap.add_argument("--rl_G", type=int, default=8, help="sampled top-5 lists per segment (group size)")
ap.add_argument("--rl_beta", type=float, default=0.05, help="KL weight to the supervised starting point")
ap.add_argument("--rl_alpha", type=float, default=0.5, help="class weight = count^-alpha (normalized), for bal rewards")
ap.add_argument("--save_val", default="", help="dir: save each row's validation log-probs per seed (for fusion with TempAgg)")
ap.add_argument("--curve", action="store_true", help="print the full LA curve (recall / top-1 / top-5 acc per tau) on validation")
ap.add_argument("--la", default="0,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5", help="logit-adjustment strengths; picked on held-out TRAIN")
ap.add_argument("--fuse", default="", help='late fusion of val predictions: "rowA|rowB,..." (sum of log-probs)')
ap.add_argument("--text", default="", help="text.pt from 11_text_feats.py: enables +hX (true text) / +hXP (predicted text)")
ap.add_argument("--dino", default="", help="dino.pt from 16_dino_grid.py: rows dino_* / both_* (plain-ViT frame tokens, group d_rec)")
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
GROUPS = {"q_last": 0, "q_hist": 1, "q_rec": 2, "t_hist": 3, "t_rec": 4, "t_fut": 5, "h": 6, "th": 7, "p_past": 8, "p_fut": 9, "d_rec": 10}
ROWS = {"q_last": ["q_last"], "q_rec": ["q_rec"], "q_hist": ["q_hist", "q_rec"], "tsm_hist": ["t_hist", "t_rec"],
        "tsm_kd": ["t_hist", "t_rec"], "fuse": ["q_hist", "q_rec", "t_hist", "t_rec"],
        "fuse_kd": ["q_hist", "q_rec", "t_hist", "t_rec"], "oracle": ["t_hist", "t_rec", "t_fut"], "hlab": []}
ROWS["fuse_kdH"], ROWS["tsm_kdH"] = ROWS["fuse_kd"], ROWS["tsm_kd"]     # same students, taught by oracleH
ROWS["fuse_th_kd"] = ROWS["fuse_kd"] + ["th"]
# history-length ablation (same teacher): no VLM history frames / TSM history 30 s or 10 s / both short
ROWS["fuse_noqh_kd"] = ["q_rec", "t_hist", "t_rec"]
ROWS["fuse_t30_kd"] = ["q_hist", "q_rec", "t_hist:15", "t_rec"]
ROWS["fuse_t10_kd"] = ["q_hist", "q_rec", "t_hist:5", "t_rec"]
ROWS["fuse_t10_noqh_kd"] = ["q_rec", "t_hist:5", "t_rec"]
ROWS["fuse_t30_noqh_kd"] = ["q_rec", "t_hist:15", "t_rec"]
# plain ViT (DINOv2, 3 x 3 grid of the 4 recent frames) instead of / next to the VLM frames; tsm_t10 = no frame tokens at all
ROWS["dino_t10_kd"] = ["d_rec", "t_hist:5", "t_rec"]
ROWS["both_t10_kd"] = ["q_rec", "d_rec", "t_hist:5", "t_rec"]
ROWS["tsm_t10_kd"] = ["t_hist:5", "t_rec"]
# hand pose: P = student gets the last 4 s of hand motion; F = teacher (oracleP) also sees hands past + next 2 s
ROWS["fuse_kdP"], ROWS["fuse_kdF"], ROWS["fuse_kdPF"] = ROWS["fuse_kd"] + ["p_past"], ROWS["fuse_kd"], ROWS["fuse_kd"] + ["p_past"]
ROWS["fuse_kdPA"] = ROWS["fuse_kdP"]          # A = + auxiliary output: forecast the next 2 s of hand motion (real target)
ROWS["tsm_kdP"], ROWS["tsm_kdF"], ROWS["tsm_kdPF"] = ROWS["tsm_kd"] + ["p_past"], ROWS["tsm_kd"], ROWS["tsm_kd"] + ["p_past"]          # + 2 tokens from the VLM's reasoning (13_reason.py thoughts)
# teachers: oracle = TSM past + real future; oracleH = the same + the TRUE past-action labels (privileged)
ROWS["oracleP"] = ROWS["oracle"] + ["p_past", "p_fut"]
TEACH = {"oracle": ("oracle", ROWS["oracle"]), "oracleH": ("oracle+hT8", ROWS["oracle"] + ["h"]),
         "oracleP": ("oracleP", ROWS["oracleP"])}   # oracleH: last 8 true labels
# history modes (row suffix): labels of the past actions used in (training, validation)
HMODE = {"hT": ("true", "true"), "hTP": ("true", "pred"), "hP": ("pred", "pred"), "hS": ("soft", "soft"),
         "hV": ("seg", "seg"), "hX": ("txtT", "txtT"), "hXP": ("txtP", "txtP")}   # hX / hXP: history as text (11_text_feats.py)       # hV = aligned video history: TSM features of each past action segment
NOISE = sorted({int(x) for x in re.findall(r"hN(\d+)", a.rows)})   # hN30 = true labels with 30% replaced at random


def hmode(hm):
    """'hS' / 'hS64' / 'hTP' ... -> (train source, val source) as 'src:length'; length = last N past actions (default 8)."""
    mt = re.fullmatch(r"(hXP|hXL)(\d*)-(\w+)", hm)         # text styles (12_text_styles.py)
    if mt:
        kind = "P" if mt.group(1) == "hXP" else "L"
        L = 1 if kind == "L" else int(mt.group(2) or 8)
        return (f"txt{kind}_{mt.group(3)}:{L}",) * 2
    mn = re.fullmatch(r"hN(\d+)(P?)(?:_(\d+))?", hm)      # hN40 = 40% noisy true labels; hN40P = same in training,
    if mn:                                                # PREDICTED labels at test (AAG+-style); _3 = last 3 steps
        L = int(mn.group(3) or 8)
        return (f"n{mn.group(1)}:{L}", f"pred:{L}" if mn.group(2) else f"n{mn.group(1)}:{L}")
    m = re.fullmatch(r"(hTP|hT|hP|hS|hV|hXP|hX)(\d*)", hm)
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
TDIM = [0]                     # (PCA dim + 1) of the text-style tokens, set when --texts is loaded
HDIM = [0]                     # values per hand-pose step, set when --hands is loaded
DDIM = [0]                     # DINOv2 width, set when --dino is loaded


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
    if a.text:
        TX = torch.load(a.text)
        for s in ("train", "validation"):
            for src, k in (("txtT", "true"), ("txtP", "pred")):
                x = TX[s][k].float(); x = torch.cat([x, torch.ones_like(x[..., :1])], -1)   # 4097-d: marks text tokens
                out[s]["txt_" + src] = x.half().to(dev)
            out[s]["txt_m"] = TX[s]["mask"].to(dev)
    if a.dino:
        DB = torch.load(a.dino)
        DDIM[0] = DB["train"]["rec"].shape[-1]
        assert DDIM[0] not in (4096, 2048, PSUM, HDIM[0], TDIM[0]), "DINO width clashes with another token type"
        md, sd = stats(DB["train"]["rec"])
        for s in ("train", "validation"):
            assert torch.equal(DB[s]["y"], Q[s]["y"]), f"{s}: DINO and VLM banks differ"
            n = len(DB[s]["y"])
            out[s]["d_rec"] = ((DB[s]["rec"].float() - md) / sd).half().reshape(n, -1, DDIM[0]).to(dev)
    if a.hands:
        HB = torch.load(a.hands)
        HDIM[0] = HB["train"]["past"].shape[-1]
        for s in ("train", "validation"):
            assert len(HB[s]["past"]) == len(Q[s]["y"]), f"{s}: hand bank and VLM bank differ"
            out[s]["p_past"], out[s]["p_fut"] = HB[s]["past"].to(dev), HB[s]["fut"].to(dev)
    if a.thoughts:
        for s in ("train", "validation"):
            th = torch.zeros(len(Q[s]["y"]), 2, 4098, dtype=torch.float16)
            for pth in glob.glob(os.path.join(a.thoughts, f"{s}_*.pt")):
                c = torch.load(pth)
                x = c["feats"].float(); x = torch.cat([x, torch.ones_like(x[..., :2])], -1)      # 4098-d marks thought tokens
                th[torch.tensor(c["ids"])] = x.half()
            out[s]["th"] = th.to(dev)
            print(f"thought tokens {s}: {int((th[:, 0, -1] > 0).sum())} segments", flush=True)
    for item in [x for x in a.texts.split(",") if x]:
        st, path = item.split("=")
        TS = torch.load(path)
        for s in ("train", "validation"):
            for k, key in (("pred", "P"), ("last", "L")):
                x = TS[s][k].float(); x = torch.cat([x, torch.ones_like(x[..., :1])], -1)    # (dim+1)-d marks style text
                out[s][f"txt{key}_{st}"] = x.half().to(dev)
            out[s][f"txtm_{st}"] = TS[s]["mask"].to(dev)
        TDIM[0] = TS["train"]["pred"].shape[-1] + 1
    cache.clear(); cache[layer] = out
    return out


def tokens(D, groups, idx, hsrc="true:8"):
    xs, ms, kinds, posi = [], [], [], []
    for g in groups:
        if g == "th":
            x = D["th"][idx]; n = x.shape[1]
            xs.append(x); ms.append(torch.ones(len(idx), n, dtype=torch.bool, device=dev))
            kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(n, device=dev))
            continue
        if g == "h":
            src, L = hsrc.split(":"); L = int(L)
            if src.startswith("txt") and "_" in src:               # text styles
                x = D[src][idx][:, -L:]
                m = D["txtm_" + src.split("_", 1)[1]][idx][:, -L:] if src.startswith("txtP") else \
                    torch.ones(len(idx), 1, dtype=torch.bool, device=dev)
                n = x.shape[1]; xs.append(x); ms.append(m)
                kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(n, device=dev))
                continue
            if src.startswith("txt"):
                x, m = D["txt_" + src][idx][:, -L:], D["txt_m"][idx][:, -L:]
                n = x.shape[1]; xs.append(x); ms.append(m)
                kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(n, device=dev))
                continue
            if src in ("soft", "seg"):
                H = D["h_idx"][idx][:, -L:]
                x = D["prob" if src == "soft" else "seg"][H.clamp(min=0)] * (H >= 0)[..., None]
            else:
                x = D["h_" + src][idx][:, -L:]
            m = D["h_m"][idx][:, -L:]
        else:
            g, _, keep = g.partition(":")                      # "t_hist:5" = only the 5 most recent history windows
            x = D[g][idx]
            m = D[g + "_m"][idx] if g + "_m" in D else torch.ones(len(idx), x.shape[1], dtype=torch.bool, device=dev)
            N = x.shape[1]
            if keep:
                x, m = x[:, -int(keep):], m[:, -int(keep):]
            n = x.shape[1]
            xs.append(x); ms.append(m)                           # positions keep their place in the full window
            kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(N - n, N, device=dev))
            continue
        n = x.shape[1]
        xs.append(x)
        ms.append(m)
        kinds.append(torch.full((n,), GROUPS[g], device=dev)); posi.append(torch.arange(n, device=dev))
    return xs, torch.cat(ms, 1), torch.cat(kinds), torch.cat(posi)


class Probe(nn.Module):
    def __init__(self, aux=False):
        super().__init__()
        d = a.d
        self.inq = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d))
        self.int = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
        self.hemb = nn.ModuleList(nn.Embedding(n + 1, d) for n in SIZES)
        self.intx = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d))                # history as text (VLM language states)
        self.inth = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d))                # reasoning (thought) tokens
        self.indino = nn.Sequential(nn.LayerNorm(DDIM[0]), nn.Linear(DDIM[0], d)) if DDIM[0] else None   # DINOv2 tokens
        self.inhand = nn.Sequential(nn.LayerNorm(HDIM[0]), nn.Linear(HDIM[0], d)) if HDIM[0] else None   # hand-pose tokens
        self.aux = aux
        if aux:                                                                          # hand forecasting: 4th query
            self.qa = nn.Parameter(torch.randn(1, 1, d) * 0.02); self.outa = nn.Linear(d, 8 * HDIM[0])
        self.intx2 = nn.Sequential(nn.LayerNorm(TDIM[0] - 1), nn.Linear(TDIM[0] - 1, d)) if TDIM[0] else None   # text styles (PCA)          # past action / verb / object labels
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
        if DDIM[0] and x.shape[-1] == DDIM[0]:
            return self.indino(x.float())
        if HDIM[0] and x.shape[-1] == HDIM[0]:
            return self.inhand(x.float())
        if x.shape[-1] == 4098:
            return self.inth(x[..., :4096].float())
        if TDIM[0] and x.shape[-1] == TDIM[0]:
            return self.intx2(x[..., :-1].float())
        if x.shape[-1] == 4097:
            return self.intx(x[..., :4096].float())
        if x.shape[-1] == PSUM:                                         # soft labels: expected label embedding
            parts = x.float().split(SIZES, -1)
            return sum(p @ e.weight[:n] for p, e, n in zip(parts, self.hemb, SIZES))
        return (self.inq if x.shape[-1] == 4096 else self.int)(x.float())

    def forward(self, xs, mask, kinds, posi, want_aux=False):
        h = torch.cat([self.embed(x) for x in xs], 1)
        h = h + self.kind(kinds) + self.pos(posi)
        pad = ~mask
        h = self.norm(self.blocks(h, src_key_padding_mask=pad))
        qs = self.q.expand(len(h), -1, -1)
        if self.aux:
            qs = torch.cat([qs, self.qa.expand(len(h), -1, -1)], 1)
        q, _ = self.xattn(qs, h, h, key_padding_mask=pad)
        q = self.drop(self.qnorm(q))
        outs = [self.outs[j](q[:, j]) for j in range(3)]
        if want_aux:
            return outs, self.outa(q[:, 3]).view(len(h), 8, HDIM[0])
        return outs


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


def plackett_luce(logits, idx):
    """log-prob of ordered picks idx [B, G, k] (without replacement) under softmax(logits [B, C])."""
    B, G, k = idx.shape
    l = logits[:, None, :].expand(B, G, -1)
    taken = torch.zeros_like(l, dtype=torch.bool)
    lp = 0
    for j in range(k):
        c = idx[..., j]
        lp = lp + l.gather(-1, c[..., None]).squeeze(-1) - l.masked_fill(taken, float("-inf")).logsumexp(-1)
        taken = taken.scatter(-1, c[..., None], True)
    return lp


def grpo(m, D, groups, htr, fi, ho, teacher, variant, seed):
    """GRPO on the probe's ACTION head: sample G top-5 lists per segment (Gumbel top-k), reward each, advantage =
    reward standardized within its group; loss = -A * log PL(list) + beta * KL(policy || start), plus the usual focal
    loss on verb / object so those heads do not drift. Epoch kept by held-out mean recall (epoch 0 = start)."""
    torch.manual_seed(1000 + seed)
    ref = copy.deepcopy(m).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=a.rl_lr, weight_decay=0.0)
    ytr = D["train"]["y"]
    w = (CNT_A.clamp(min=1) ** -a.rl_alpha); w = w / w[ytr[fi, 0]].mean()          # mean weight 1 over training segments
    h0 = scores(predict(m, D["train"], groups, ho, htr), ytr[ho]); best = ((h0["verb"] + h0["noun"] + h0["action"]) / 3, copy.deepcopy(m.state_dict()), 0)
    for ep in range(a.rl_epochs):
        m.train(); stats = []
        for b in fi[torch.randperm(len(fi), device=dev)].split(a.bs):
            lg = m(*tokens(D["train"], groups, b, htr))
            la = lg[0].float()
            gum = -torch.log(-torch.log(torch.rand(len(b), a.rl_G, la.shape[-1], device=dev).clamp(1e-9, 1 - 1e-9)))
            idx = (la.detach()[:, None, :] + gum).topk(5, -1).indices                     # [B, G, 5]
            y = ytr[b]
            hit = (idx == y[:, None, None, 0]).any(-1).float()
            r = hit.clone()
            if variant in ("bal", "sbal", "tbal"):
                r = r * w[y[:, 0]][:, None]
            if variant == "sbal":
                vh = (A2V[idx] == y[:, None, None, 1]).any(-1).float(); oh = (A2O[idx] == y[:, None, None, 2]).any(-1).float()
                r = r + w[y[:, 0]][:, None] * 0.25 * (vh + oh) * (1 - hit)
            if variant == "tbal":
                with torch.no_grad():
                    tp = teacher[0](*tokens(D["train"], teacher[1], b, teacher[2]))[0].float().softmax(-1)
                r = r + tp[:, None, :].expand(-1, a.rl_G, -1).gather(-1, idx).sum(-1)
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            lp = plackett_luce(la, idx)
            with torch.no_grad():
                lr_ = ref(*tokens(D["train"], groups, b, htr))[0].float().log_softmax(-1)
            ls = la.log_softmax(-1)
            kl = (ls.exp() * (ls - lr_)).sum(-1).mean()
            loss = -(adv * lp).mean() + a.rl_beta * kl + focal(lg[1], y[:, 1]) + focal(lg[2], y[:, 2])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            stats.append((r.mean().item(), hit.mean().item(), kl.item()))
        h = scores(predict(m, D["train"], groups, ho, htr), ytr[ho]); sel = (h["verb"] + h["noun"] + h["action"]) / 3
        rm, hm_, km = (sum(x[i] for x in stats) / len(stats) for i in range(3))
        print(f"    [GRPO {variant} seed {seed}] epoch {ep + 1}  reward {rm:.3f}  hit {hm_:.3f}  KL {km:.4f}  held-out mean recall {sel:5.1f}"
              f"{' *' if sel > best[0] else ''}", flush=True)
        if sel > best[0]:
            best = (sel, copy.deepcopy(m.state_dict()), ep + 1)
    m.load_state_dict(best[1])
    return best[2]


def hand_forecast_loss(pred, target):
    """MSE on the next 2 s of hand motion, only where a hand is present (last value of each hand's block)."""
    t = target.float().view(len(target), 8, 2, HDIM[0] // 2); p = pred.view_as(t)
    w = t[..., -1:]                                        # present flag per step and hand
    return (((p - t) ** 2) * w).sum() / (w.sum() * t.shape[-1]).clamp(min=1)


def train(name, base, D, seed, fi, ho, teacher=None, rlv=""):
    torch.manual_seed(seed)
    base, _, hm = base.partition("+")
    use_aux = base.endswith("PA")
    groups = ROWS[base] + (["h"] if hm else [])
    htr, hte = hmode(hm) if hm else ("true:8", "true:8")
    m = Probe(aux=use_aux).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * math.ceil(len(fi) / a.bs), pct_start=0.1)
    best = (-1.0, None, 0)
    for ep in range(a.epochs):
        m.train()
        perm = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            if use_aux:
                lg, fc = m(*tokens(D["train"], groups, b, htr), want_aux=True)
            else:
                lg = m(*tokens(D["train"], groups, b, htr))
            loss = sum(focal(lg[j], D["train"]["y"][b, j]) for j in range(3))
            if use_aux:
                loss = loss + a.lam_hand * hand_forecast_loss(fc, D["train"]["p_fut"][b])
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
    rl_ep = grpo(m, D, groups, htr, fi, ho, teacher, rlv, seed) if rlv else None
    lg = predict(m, D["validation"], groups, hsrc=hte)
    VAL[name] = [F.log_softmax(o, -1).half() for o in lg]                 # kept for late fusion
    HO[name] = [F.log_softmax(o, -1).half() for o in predict(m, D["train"], groups, ho, hte)]
    r = scores(lg, D["validation"]["y"]); r["epoch"] = best[2]
    return r, m


rows = a.rows.split(",")
FUSE = [f.split("|") for f in a.fuse.split(",") if f]
TAUS = [float(t) for t in a.la.split(",") if t]
HO = {}
_cnt = [torch.bincount(Q["train"]["y"][:, j], minlength=n).float() + 1 for j, n in enumerate(SIZES)]
LOGP = [(c / c.sum()).log().to(dev) for c in _cnt]                         # log class priors (train)
CNT_A = torch.bincount(Q["train"]["y"][:, 0], minlength=SIZES[0]).float().to(dev)      # action counts (GRPO class weights)
_yall = torch.cat([Q["train"]["y"], Q["validation"]["y"]])
A2V = torch.zeros(SIZES[0], dtype=torch.long); A2O = torch.zeros(SIZES[0], dtype=torch.long)
A2V[_yall[:, 0]], A2O[_yall[:, 0]] = _yall[:, 1], _yall[:, 2]
A2V, A2O = A2V.to(dev), A2O.to(dev)                                            # action -> verb / object (structured reward)


def logit_adjust(val_lp, ho_lp, y_ho, y_val):
    """post-hoc logit adjustment: score - tau * log prior; tau picked on the held-out TRAIN split (mean recall)."""
    best = None
    for t in TAUS:
        h = scores([ho_lp[j].float() - t * LOGP[j] for j in range(3)], y_ho)
        sel = (h["verb"] + h["noun"] + h["action"]) / 3
        if best is None or sel > best[0]:
            best = (sel, t)
    r = scores([val_lp[j].float() - best[1] * LOGP[j] for j in range(3)], y_val); r["tau"] = best[1]
    return r


CURVES = collections.defaultdict(lambda: collections.defaultdict(list))   # name -> tau -> [scores per seed]


def curve(name, val_lp, y_val):
    for t in TAUS:
        CURVES[name][t].append(scores([val_lp[j].float() - t * LOGP[j] for j in range(3)], y_val))
res = collections.defaultdict(list)
VAL, TIMES, t_start = {}, {}, time.time()
def _teacher_of(r_):
    b = r_.split("@")[0].split("#")[0].split("+")[0]
    if not b.endswith(("_kd", "_kdH", "_kdP", "_kdF", "_kdPF", "_kdPA")):
        return None
    return "oracleH" if b.endswith("_kdH") else "oracleP" if b.endswith(("_kdF", "_kdPF")) else "oracle"


JOBS = rows + sorted({_teacher_of(r_) for r_ in rows} - {None})


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
    if a.train_subset:
        S = torch.tensor(json.load(open(a.train_subset)))
        perm = S[torch.randperm(len(S), generator=torch.Generator().manual_seed(seed))].to(dev)
    ho, fi = perm[: len(perm) // 10], perm[len(perm) // 10:]
    teachers = {}
    for name in rows:
        base, layer = (name.split("@") + ["27"])[:2]
        base, _, rlv = base.partition("#")                  # fuse_kd#bal = fuse_kd + GRPO with the 'bal' reward
        D = gpu_data(int(layer))
        b0 = base.split("+")[0]
        if b0.endswith(("_kd", "_kdH", "_kdP", "_kdF", "_kdPF", "_kdPA")):
            tk = "oracleH" if b0.endswith("_kdH") else "oracleP" if b0.endswith(("_kdF", "_kdPF")) else "oracle"
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
            r, _ = train(name, base, D, seed, fi, ho, teacher=teachers[tk], rlv=rlv)
        else:
            t_ = time.time()
            r, _ = train(name, base, D, seed, fi, ho, rlv=rlv)
        TIMES.setdefault(name, time.time() - t_); done_names.append(name)
        if a.save_val:
            os.makedirs(a.save_val, exist_ok=True)
            torch.save([x.cpu() for x in VAL[name]], os.path.join(a.save_val, f"{name.replace('|', '_')}_seed{seed}.pt"))
        res[name].append(r)
        print(f"{name:12s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  epoch {r['epoch']}", flush=True)
        if a.curve:
            curve(name, VAL[name], D["validation"]["y"])
        if TAUS:
            rl = logit_adjust(VAL[name], HO[name], D["train"]["y"][ho], D["validation"]["y"]); res[name + " +LA"].append(rl)
            print(f"{name + ' +LA':12s} seed {seed}: verb {rl['verb']:5.1f}  object {rl['noun']:5.1f}  action {rl['action']:4.1f}"
                  f"  top-1 {rl['action_top1']:4.1f}  tau {rl['tau']}", flush=True)
        print(eta(seed, done_names), flush=True)
    for A, B in FUSE:
        fl = [VAL[A][j].float() + VAL[B][j].float() for j in range(3)]
        r = scores(fl, D["validation"]["y"]); res[f"{A}|{B}"].append(r)
        if a.save_val:
            torch.save([x.half().cpu() for x in fl], os.path.join(a.save_val, f"{A}_{B}_seed{seed}.pt"))
        print(f"{A + '|' + B:24s} seed {seed}: verb {r['verb']:5.1f}  object {r['noun']:5.1f}  action {r['action']:4.1f}"
              f"  top-1 {r['action_top1']:4.1f}  (late fusion)", flush=True)
        if a.curve:
            curve(f"{A}|{B}", fl, D["validation"]["y"])
        if TAUS:
            fh = [HO[A][j].float() + HO[B][j].float() for j in range(3)]
            rl = logit_adjust(fl, fh, D["train"]["y"][ho], D["validation"]["y"]); res[f"{A}|{B} +LA"].append(rl)
            print(f"{A + '|' + B + ' +LA':24s} seed {seed}: verb {rl['verb']:5.1f}  object {rl['noun']:5.1f}  action {rl['action']:4.1f}"
                  f"  top-1 {rl['action_top1']:4.1f}  tau {rl['tau']}", flush=True)

print(f"\n({a.seeds} seeds, mean +- sd; class-mean top-5 recall on validation_challenge, camera C10119)")
for name in [t for t in TEACH if t in res] + [x for r_ in rows + [f"{A}|{B}" for A, B in FUSE] for x in (r_, r_ + " +LA") if x in res]:
    rs = res[name]
    ms = lambda k: (lambda t: f"{t.mean():5.1f} +- {t.std():3.1f}")(torch.tensor([r[k] for r in rs]))   # noqa: E731
    print(f"{name:12s} verb {ms('verb')} | object {ms('noun')} | action {ms('action')} | action top-1 {ms('action_top1')}"
          f" | action top-5 acc {ms('action_top5')}")
for name, cv in CURVES.items():
    print(f"\nLA curve on validation: {name} ({a.seeds} seeds, mean)\n  tau | verb  object  action | action top-1  top-5 acc")
    for t, rs in cv.items():
        m = lambda k: sum(r[k] for r in rs) / len(rs)   # noqa: E731
        print(f"{t:5.2f} | {m('verb'):5.1f}  {m('noun'):5.1f}  {m('action'):5.1f}  | {m('action_top1'):5.1f}  {m('action_top5'):5.1f}")
json.dump({"args": vars(a), "results": res, "curves": {k: {str(t): v for t, v in cv.items()} for k, cv in CURVES.items()}}, open(os.path.join(a.q3, f"probe{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)
