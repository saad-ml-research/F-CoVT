"""F-CoVT core: future thought tokens, heads, frozen experts and losses.

Two modes share the same thought-slot mechanism and differ ONLY in what the slots are supervised with:
  fcovt          (row E) per future step: <start> seg x8, depth x4, dino x4 <end>
                         seg   -> SAM prompts -> frozen SAM decoder on the CURRENT frame's SAM embedding
                                  -> masks, Hungarian Dice+BCE vs SAM masks of the FUTURE frame (precomputed)
                         depth -> 4 tokens dotted with DepthAnything features of the CURRENT frame
                                  -> depth map, L1 vs DepthAnything depth of the FUTURE frame
                         dino  -> cross-attn queries -> 1025 x 1024, MSE vs DINOv2 features of the FUTURE frame
  fcovt_generic  (row D) per future step: <start> lat x16 <end>
                         -> cross-attn queries -> 64 x H, MSE vs Qwen's own (frozen) vision features of the
                            FUTURE frame, pooled to 64 tokens and layer-normalised (Future-L1-style generic target)

Heads follow CoVT (Qin et al.) train/src/training/covt_qwen2_5_vl.py; loss scales copied from CoVT's AnchorLoss.
The future frame is never an input: only the current/context frames + fixed thought tokens are fed to the VLM.
The hidden states at the thought positions are what the heads read, so at test time the thought tokens can be
force-appended to the prompt (FCOVT_PREFIX) and the model computes its own thoughts in the prefill.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

COVT_SRC = os.environ.get("COVT_SRC", "/workspace/CoVT/train/src")
MAX_FUTURE = int(os.environ.get("FCOVT_MAX_FUTURE", "2"))
GAMMA = float(os.environ.get("FCOVT_GAMMA", "1.0"))
W_SEG = float(os.environ.get("FCOVT_W_SEG", "1.0"))     # per-expert weights (for ablations / gradient checks)
W_DEP = float(os.environ.get("FCOVT_W_DEPTH", "1.0"))
W_DINO = float(os.environ.get("FCOVT_W_DINO", "1.0"))

START, END = "<|fcovt_start|>", "<|fcovt_end|>"
SEG, DEP, DINO, LAT = "<|fcovt_seg|>", "<|fcovt_depth|>", "<|fcovt_dino|>", "<|fcovt_lat|>"
ALL_TOKENS = [START, END, SEG, DEP, DINO, LAT]
N_SEG, N_DEP, N_DINO, N_LAT = 8, 4, 4, 16
N_GENERIC_Q = 64


# ----------------------------------------------------------------------------- text

def block(mode):
    body = SEG * N_SEG + DEP * N_DEP + DINO * N_DINO if mode == "fcovt" else LAT * N_LAT
    return START + body + END


def thought_text(mode, n_future):
    """Thought tokens for n_future future steps (capped at MAX_FUTURE, at least 1)."""
    return "".join(block(mode) for _ in range(max(1, min(int(n_future), MAX_FUTURE))))


# Input-embedding init for the thought tokens (rows stay frozen). "mean" (rows D/E): the average of all real
# token rows -- an unusually short vector that confused the model (row F smoke: answer CE 7-19 vs 2-3 without
# the tokens). "word": each token starts as a related real word, like CoVT's named tokens.
INIT_WORDS = {START: " [", END: " ]", SEG: " segmentation", DEP: " depth", DINO: " feature", LAT: " latent"}


def init_thought_embeddings(tok, emb, latent_id, how=None):
    how = how or os.environ.get("FCOVT_EMB_INIT", "mean")
    with torch.no_grad():
        for t in ALL_TOKENS:
            i = tok.convert_tokens_to_ids(t)
            if how == "word":
                w = tok(INIT_WORDS[t], add_special_tokens=False)["input_ids"]
                emb[i] = emb[w].float().mean(0).to(emb.dtype)
            else:
                emb[i] = emb[:latent_id].float().mean(0).to(emb.dtype)
    return how


# ----------------------------------------------------------------------------- heads

class ExpertHeads(nn.Module):
    """CoVT's SAM / depth / DINO heads."""

    def __init__(self, hidden=3584):
        super().__init__()
        self.sam_proj = nn.Linear(hidden, 256)
        self.sam_q = nn.Parameter(torch.randn(N_SEG, 256))
        self.sam_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.depth_gen = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1024))
        self.dino_proj = nn.Linear(hidden, 1024)
        self.dino_q = nn.Parameter(torch.randn(1025, 1024))
        self.dino_attn = nn.MultiheadAttention(1024, 8, batch_first=True)

    def sam_prompts(self, h):                       # h [8, H] -> [8, 256]
        kv = F.normalize(self.sam_proj(h), dim=-1)[None]
        out, _ = self.sam_attn(self.sam_q[None].to(kv.dtype), kv, kv)
        return out[0]

    def depth_tokens(self, h):                      # h [4, H] -> [4, 1024]
        return self.depth_gen(h)

    def dino_pred(self, h):                         # h [4, H] -> [1025, 1024]
        kv = F.normalize(self.dino_proj(h), dim=-1)[None]
        out, _ = self.dino_attn(self.dino_q[None].to(kv.dtype), kv, kv)
        return out[0]


class GenericHead(nn.Module):
    """Same style (proj + learned queries + cross-attn), generic target: Qwen's own vision features."""

    def __init__(self, hidden=3584):
        super().__init__()
        self.proj = nn.Linear(hidden, 1024)
        self.q = nn.Parameter(torch.randn(N_GENERIC_Q, 1024))
        self.attn = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.out = nn.Linear(1024, hidden)

    def forward(self, h):                           # h [16, H] -> [64, H]
        kv = F.normalize(self.proj(h), dim=-1)[None]
        o, _ = self.attn(self.q[None].to(kv.dtype), kv, kv)
        return self.out(o[0])


class StagedHeads(ExpertHeads):
    """Row F (CoVT curriculum): expert heads + trainable OUTPUT rows for the 6 thought tokens, so the model
    can write the thoughts itself (CoVT trains the new lm_head rows). Input-embedding rows stay frozen."""

    def __init__(self, hidden=3584):
        super().__init__(hidden)
        self.out_rows = nn.Parameter(torch.zeros(len(ALL_TOKENS), hidden))


def build_heads(mode, hidden):
    if mode == "staged":
        return StagedHeads(hidden)
    return ExpertHeads(hidden) if mode == "fcovt" else GenericHead(hidden)


# ----------------------------------------------------------------------------- row F: CoVT's 4-stage curriculum
# CoVT train/scripts/finetune.sh: 6k steps tokens-in-prompt, then 3k "What is the <expert> of the image?",
# 3k "<think> Because the <expert> of the image is <pads>. </think><answer>", 4k of which 1/6 have no thoughts
# and a random subset of experts, expert losses off for the last 2k. Same proportions of our step budget:
STAGES = [float(x) for x in os.environ.get("FCOVT_STAGES", "0.375,0.5625,0.75,0.875").split(",")]
S1_TARGET = os.environ.get("FCOVT_S1_TARGET", "current")   # stage 1 targets: current frame (CoVT-exact) or future
STEP_FILE = os.environ.get("FCOVT_STEP_FILE", "")
EXPERTS = [("seg", SEG, N_SEG, "segmentation"), ("depth", DEP, N_DEP, "depth map"),
           ("dino", DINO, N_DINO, "perception feature")]      # CoVT's order and names (data.get_anchor_task_name)


def write_step(step, max_steps):
    """Trainer -> dataloader workers: the current step (workers are separate processes)."""
    if STEP_FILE:
        os.makedirs(os.path.dirname(STEP_FILE) or ".", exist_ok=True)
        with open(STEP_FILE + ".tmp", "w") as f:
            f.write(f"{step} {max_steps}")
        os.replace(STEP_FILE + ".tmp", STEP_FILE)


def current_stage():
    """-> (stage 1..4, expert losses on?). Without a step file: stage 3 with experts."""
    try:
        s, m = open(STEP_FILE).read().split()
        frac = int(s) / max(1, int(m))
    except Exception:
        return 3, True
    return 1 + sum(frac >= b for b in STAGES[:3]), frac < STAGES[3]


def _pad(i):
    return START + EXPERTS[i][1] * EXPERTS[i][2] + END


def sample_plan(stage, experts_on, rng):
    """Per-sample choices, made in the collator so the text and the loss agree."""
    if os.environ.get("FCOVT_DBG_PLAIN"):           # diagnostic: B' format (answer only) through the row F code
        return {"stage": stage, "experts": [], "loss": False, "target": "fut"}
    if stage == 4 and rng.randint(0, 5) == 0:       # CoVT: 1 in 6 answers without thoughts
        return {"stage": 4, "experts": [], "loss": False, "target": "fut"}
    ex = sorted(rng.sample(range(3), rng.randint(1, 3))) if stage == 4 else [0, 1, 2]
    target = "cur" if (stage == 1 and S1_TARGET == "current") else "fut"
    return {"stage": stage, "experts": ex, "loss": bool(experts_on), "target": target}


def staged_texts(plan, answer):
    """-> (pads to put right after the current frame in the prompt, question override or None, response)."""
    st, ex = plan["stage"], plan["experts"]
    if st == 1:
        return "".join(_pad(i) for i in ex), None, f"<answer>{answer}</answer>"
    if st == 2:
        names = ", ".join(EXPERTS[i][3] for i in ex)
        return "", f"What is the {names} of the future frame?", "".join(_pad(i) for i in ex) + "\n"
    if not ex:
        return "", None, f"<answer>{answer}</answer>"
    parts = [f"the {EXPERTS[i][3]} of the future frame is {_pad(i)}" for i in ex]
    s = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    return "", None, f"<think> Because {s}. </think>\n<answer> {answer} </answer>"


# ----------------------------------------------------------------------------- losses (from CoVT)

def dice_coeff(p, g, eps=1e-6):
    p = p.sigmoid()
    inter = (p * g).sum((-2, -1))
    union = p.sum((-2, -1)) + g.sum((-2, -1))
    return (2 * inter + eps) / (union + eps)


def sam_token_loss(pred, gt):
    """CoVT get_sam_token_loss: Hungarian match on 1-Dice, Dice+BCE on matches, 0.1*BCE-to-empty on the rest."""
    if gt.shape[0] == 0:
        return F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred))
    with torch.no_grad():
        cost = 1.0 - dice_coeff(pred.unsqueeze(1), gt.unsqueeze(0))
    r, c = linear_sum_assignment(cost.float().cpu().numpy())
    r = torch.as_tensor(r, device=pred.device)
    c = torch.as_tensor(c, device=pred.device)
    pos_p, pos_g = pred[r], gt[c]
    loss = (1.0 - dice_coeff(pos_p, pos_g).mean()) + F.binary_cross_entropy_with_logits(pos_p, pos_g)
    un = [i for i in range(pred.shape[0]) if i not in set(r.tolist())]
    if un:
        neg = pred[un]
        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    return loss


# ----------------------------------------------------------------------------- frozen experts

class Experts:
    def __init__(self, device):
        if COVT_SRC not in sys.path:
            sys.path.insert(0, COVT_SRC)
        from anchors.segment_anything import sam_model_registry, SamPredictor
        from anchors.segment_anything.utils.transforms import ResizeLongestSide
        from anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

        self.device = device
        sam_ck = os.path.join(COVT_SRC, "anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_ck).to(device).eval()
        self.sam_pred = SamPredictor(self.sam)
        self.sam_input_size = ResizeLongestSide(self.sam.image_encoder.img_size).apply_image(
            np.zeros((256, 256, 3), np.uint8)).shape[:2]

        dep_ck = os.path.join(COVT_SRC, "anchors/DepthAnything/ckpt/depth_anything_v2_vitl.pth")
        self.depth = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        self.depth.load_state_dict(torch.load(dep_ck, map_location="cpu"))
        self.depth = self.depth.to(device).eval()

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        for m in (self.sam, self.depth, self.dino):
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def sam_embed(self, img):                       # current frame -> [1, 256, 64, 64]
        self.sam_pred.set_image(np.array(img.resize((256, 256))))
        return self.sam_pred.get_image_embedding()

    def sam_decode(self, emb, prompts):             # prompts [n, 256] (grad) -> mask logits [n, 256, 256]
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None,
                                                text_embeds=prompts.float().unsqueeze(1))
        low, _ = self.sam.mask_decoder(image_embeddings=emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                       sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                       multimask_output=False)
        up = self.sam.postprocess_masks(low, input_size=self.sam_input_size, original_size=(256, 256))
        return up[:, 0]

    @torch.no_grad()
    def depth_feats(self, img):
        """-> 4 patch-feature maps [N, 1024], (ph, pw), depth map [1, 1, 256, 256] (CoVT layers 4/11/17/23)."""
        import cv2
        raw = cv2.cvtColor(np.array(img.resize((256, 256))), cv2.COLOR_RGB2BGR)  # image2tensor expects BGR
        x, (h, w) = self.depth.image2tensor(raw)
        x = x.to(self.device)
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        feats = self.depth.pretrained.get_intermediate_layers(x, [4, 11, 17, 23], return_class_token=True)
        d = F.relu(self.depth.depth_head(feats, ph, pw))
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=True)
        return [f[0][0] for f in feats], (ph, pw), d

    @staticmethod
    def depth_from_tokens(tokens, feats, phw):      # tokens [4, 1024] (grad) -> [1, 1, 256, 256]
        ph, pw = phw
        maps = []
        for i in range(4):
            s = (feats[i].float() @ tokens[i].float()).view(1, 1, ph, pw)
            maps.append(F.interpolate(s, size=(256, 256), mode="bilinear", align_corners=False))
        return torch.cat(maps, 1).mean(1, keepdim=True)

    @torch.no_grad()
    def dino_feats(self, img):                      # -> [1025, 1024] (CLS + 32x32 patches, final norm)
        im = img.resize((448, 448), Image.BICUBIC)
        x = torch.from_numpy(np.array(im)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        out = self.dino.forward_features((x - self.dino_mean) / self.dino_std)
        return torch.cat([out["x_norm_clstoken"][:, None], out["x_norm_patchtokens"]], 1)[0]


# ----------------------------------------------------------------------------- loss module

def _load_img(frames_dir, k):
    return Image.open(os.path.join(frames_dir, f"f{k}.jpg")).convert("RGB")


def _load_masks(frames_dir, k, device):
    p = os.path.join(frames_dir.replace("/frames/", "/sam_masks/"), f"f{k}.npz")
    z = np.load(p)
    m = np.unpackbits(z["masks"], axis=-1)[..., :256] if int(z["n"]) > 0 else np.zeros((0, 256, 256), np.uint8)
    return torch.from_numpy(m.astype(np.float32)).to(device)


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


class FCoVTLoss:
    def __init__(self, mode, heads, processor):
        self.mode = mode
        self.heads = heads
        self.processor = processor
        tok = processor.tokenizer
        self.ids = {t: tok.convert_tokens_to_ids(t) for t in ALL_TOKENS}
        self.experts = None

    def _blocks(self, ids):
        s = (ids == self.ids[START]).nonzero().flatten().tolist()
        e = (ids == self.ids[END]).nonzero().flatten().tolist()
        return list(zip(s, e))

    def __call__(self, model, hidden, input_ids, metas):
        """hidden [B, L, H] (last layer), input_ids [B, L], metas: list of {frames_dir, cur, fut}."""
        dev = hidden.device
        parts = {}
        total = hidden.new_zeros(())
        n_steps = 0
        if self.mode == "fcovt" and self.experts is None:
            self.experts = Experts(dev)
        for b, meta in enumerate(metas):
            ids = input_ids[b]
            blocks = self._blocks(ids)[: len(meta["fut"])]
            if not blocks:
                continue
            cur = _load_img(meta["frames_dir"], meta["cur"])
            if self.mode == "fcovt":
                cur_emb = self.experts.sam_embed(cur)
                cur_feats, phw, _ = self.experts.depth_feats(cur)
            for (s, e), k in zip(blocks, meta["fut"]):
                h = hidden[b, s + 1:e]
                t = ids[s + 1:e]
                fut = _load_img(meta["frames_dir"], k)
                if self.mode == "fcovt":
                    pm = self.experts.sam_decode(cur_emb, self.heads.sam_prompts(h[t == self.ids[SEG]]))
                    l_seg = sam_token_loss(pm, _load_masks(meta["frames_dir"], k, dev))
                    _, _, d_gt = self.experts.depth_feats(fut)
                    d_pred = self.experts.depth_from_tokens(self.heads.depth_tokens(h[t == self.ids[DEP]]), cur_feats, phw)
                    l_dep = F.l1_loss(d_pred, d_gt) * 0.005
                    l_dino = F.mse_loss(self.heads.dino_pred(h[t == self.ids[DINO]]).float(),
                                        self.experts.dino_feats(fut).float()) * 0.5
                    step = W_SEG * l_seg + W_DEP * l_dep + W_DINO * l_dino
                    for name, v in (("seg", l_seg), ("depth", l_dep), ("dino", l_dino)):
                        parts[f"train/{name}_loss"] = parts.get(f"train/{name}_loss", 0.0) + float(v.detach())
                else:
                    tgt = self._qwen_target(model, fut, dev)
                    pred = self.heads(h[t == self.ids[LAT]]).float()
                    step = F.mse_loss(pred, tgt) * 0.5
                    parts["train/generic_loss"] = parts.get("train/generic_loss", 0.0) + float(step.detach())
                total = total + step
                n_steps += 1
        if n_steps:
            total = total / n_steps
            parts = {k: v / n_steps for k, v in parts.items()}
        return GAMMA * total, parts

    def staged(self, hidden, input_ids, metas):
        """Row F expert losses: tokens found by type anywhere in the sequence (prompt in stage 1, response after).
        Target frame = current (stage 1, CoVT-exact) or the first future frame. Heads read the CURRENT frame's
        SAM embedding / depth features, as in rows E."""
        dev = hidden.device
        if self.experts is None:
            self.experts = Experts(dev)
        parts, total, n = {}, hidden.new_zeros(()), 0
        for b, meta in enumerate(metas):
            p = meta.get("plan") or {}
            parts["train/stage"] = float(p.get("stage", 0))
            if not p.get("loss") or not p.get("experts"):
                continue
            k = meta["cur"] if p["target"] == "cur" else (meta["fut"] or [None])[0]
            if k is None:
                continue
            ids, h = input_ids[b], hidden[b]
            cur = _load_img(meta["frames_dir"], meta["cur"])
            tgt = cur if k == meta["cur"] else _load_img(meta["frames_dir"], k)
            step, used = hidden.new_zeros(()), 0
            for i in p["experts"]:
                name, tok, num, _ = EXPERTS[i]
                hs = h[ids == self.ids[tok]]
                if hs.shape[0] != num:
                    continue
                if name == "seg":
                    mp = os.path.join(meta["frames_dir"].replace("/frames/", "/sam_masks/"), f"f{k}.npz")
                    if not os.path.exists(mp):
                        parts["train/seg_missing"] = parts.get("train/seg_missing", 0.0) + 1
                        continue
                    pm = self.experts.sam_decode(self.experts.sam_embed(cur), self.heads.sam_prompts(hs))
                    l = sam_token_loss(pm, _load_masks(meta["frames_dir"], k, dev)) * W_SEG
                elif name == "depth":
                    cf, phw, _ = self.experts.depth_feats(cur)
                    _, _, d_gt = self.experts.depth_feats(tgt)
                    l = F.l1_loss(self.experts.depth_from_tokens(self.heads.depth_tokens(hs), cf, phw), d_gt) * 0.005 * W_DEP
                else:
                    l = F.mse_loss(self.heads.dino_pred(hs).float(), self.experts.dino_feats(tgt).float()) * 0.5 * W_DINO
                parts[f"train/{name}_loss"] = parts.get(f"train/{name}_loss", 0.0) + float(l.detach())
                step, used = step + l, used + 1
            if used:
                total, n = total + step, n + 1
        if n:
            total = total / n
            parts = {k: (v / n if k.endswith("_loss") else v) for k, v in parts.items()}
        return GAMMA * total, parts

    @torch.no_grad()
    def _qwen_target(self, model, img, dev):
        base = _unwrap(model)
        ip = self.processor.image_processor(images=[img], return_tensors="pt")
        pv = ip["pixel_values"].to(dev, dtype=next(base.parameters()).dtype)
        feats = base.model.get_image_features(pv, ip["image_grid_thw"].to(dev))
        feats = torch.cat(list(feats), 0) if isinstance(feats, (list, tuple)) else feats
        pooled = F.adaptive_avg_pool1d(feats.float().T[None], N_GENERIC_Q)[0].T
        return F.layer_norm(pooled, pooled.shape[-1:])
