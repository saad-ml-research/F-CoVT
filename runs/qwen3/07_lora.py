"""Step C of the VLM ladder: LoRA fine-tuning of Qwen3-VL-8B for anticipation, with future teaching (camera C10119).
Student = our best probe (VLM grid + TSM + KD), but the VLM tokens of the 4 RECENT frames now come from a LoRA-tuned
Qwen3-VL, computed live:
  recent 4 frames -> Qwen3-VL (language layers cut at 27, LoRA rank 16 on q/k/v/o, vision frozen) -> layer-27 image
  tokens -> 3 x 3 grid per frame (36 tokens)
  + cached frozen tokens: VLM history grid (05_features_grid.py), TSM history + recent (runs/tsm/01_bank.py)
  -> probe (2 blocks + 3 queries) -> action / verb / object
Loss = focal (true next action) + lam * KD to a TSM oracle teacher (past + real future), whose logits are
precomputed once at the start. Trained: LoRA + probe. Validation is scored after every epoch (no epoch picking).
usage: python 07_lora.py /workspace/data/a101_vlm /workspace/outputs/q3_grid /workspace/outputs/tsm_bank OUT
       [--epochs 1] [--max_steps 30 (speed test)] [--val_limit 2000]
"""
import argparse, collections, glob, json, math, os, re, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("vlm"); ap.add_argument("q3"); ap.add_argument("tsm"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--grid", type=int, default=3)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr_lora", type=float, default=1e-4)
ap.add_argument("--lr_probe", type=float, default=1e-4, help="probe lr during LoRA (after the warm start)")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--warm", type=int, default=20, help="epochs of probe-only training on cached features before LoRA")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max_steps", type=int, default=0, help="speed test: stop after N optimizer steps")
ap.add_argument("--val_limit", type=int, default=0)
ap.add_argument("--lam_kd", type=float, default=2.0)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.manual_seed(a.seed)
log = open(os.path.join(a.out, "log.txt"), "a")


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()


# ------------------------------------------------------------------ cached tokens (CPU) + labels
def load_q3(split):
    parts = sorted(glob.glob(os.path.join(a.q3, f"{split}_*.pt")), key=lambda p: int(re.findall(r"_(\d+)\.pt$", p)[0]))
    ch = [torch.load(p) for p in parts]
    li = ch[0]["layers"].index(a.layer)
    return {"hist": torch.cat([c["frames"][:, :8, :, li] for c in ch]), "rec": torch.cat([c["frames"][:, 8:, :, li] for c in ch]),
            "mask": torch.cat([c["mask"][:, :8] for c in ch]), "y": torch.cat([c["y"] for c in ch])}


Q = {s: load_q3(s) for s in ("train", "validation")}
T = {s: torch.load(os.path.join(a.tsm, f"{s}.pt")) for s in ("train", "validation")}
IDX = {s: json.load(open(os.path.join(a.vlm, f"{s}.json"))) for s in ("train", "validation")}
for s in Q:
    assert torch.equal(Q[s]["y"], T[s]["y"]) and len(IDX[s]) == len(Q[s]["y"]), s
SIZES = [int(max(Q["train"]["y"][:, j].max(), Q["validation"]["y"][:, j].max())) + 1 for j in range(3)]
C = Q["train"]["hist"].shape[2]                                            # grid cells per frame


def stats(x):
    x = x[:4000].float().reshape(-1, x.shape[-1]); return x.mean(0), x.std(0) + 1e-3


MQ, SQ = stats(Q["train"]["rec"]); MT, ST = stats(T["train"]["recent"])
MQg, SQg = MQ.to(dev), SQ.to(dev)
say(f"train {len(IDX['train'])}  val {len(IDX['validation'])} | classes {SIZES} | grid cells {C}")


def cached(s, idx, vlm=True):
    """frozen tokens for a batch, standardized, on the GPU (vlm=False: TSM only, for the teacher)."""
    n = len(idx)
    D = {"q_hist": ((Q[s]["hist"][idx].float() - MQ) / SQ).reshape(n, 8 * C, -1).to(dev, non_blocking=True),
         "q_hist_m": Q[s]["mask"][idx].repeat_interleave(C, 1).to(dev)} if vlm else {}
    return D | {"t_hist": ((T[s]["hist"][idx].float() - MT) / ST).to(dev), "t_hist_m": T[s]["hist_mask"][idx].to(dev),
            "t_rec": ((T[s]["recent"][idx].float() - MT) / ST).to(dev),
            "t_fut": ((T[s]["fut"][idx].float() - MT) / ST).to(dev), "y": Q[s]["y"][idx].to(dev)}


# ------------------------------------------------------------------ probe (same as 04_probe.py)
KINDS = {"q_hist": 1, "q_rec": 2, "t_hist": 3, "t_rec": 4, "t_fut": 5}


class Probe(nn.Module):
    def __init__(self, d=512, blocks=2, drop=0.1):
        super().__init__()
        self.inq = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, d))
        self.int = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
        self.kind = nn.Embedding(8, d); self.pos = nn.Embedding(128, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, blocks)
        self.norm = nn.LayerNorm(d); self.q = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, dropout=drop, batch_first=True)
        self.qnorm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
        self.outs = nn.ModuleList(nn.Linear(d, n) for n in SIZES)

    def forward(self, groups):
        """groups: list of (name, tokens [B, n, D], mask [B, n])."""
        hs, ms = [], []
        for g, x, m in groups:
            h = (self.inq if x.shape[-1] == 4096 else self.int)(x.float())
            hs.append(h + self.kind.weight[KINDS[g]] + self.pos(torch.arange(x.shape[1], device=x.device)))
            ms.append(m)
        h, pad = torch.cat(hs, 1), ~torch.cat(ms, 1)
        h = self.norm(self.blocks(h, src_key_padding_mask=pad))
        q, _ = self.xattn(self.q.expand(len(h), -1, -1), h, h, key_padding_mask=pad)
        q = self.drop(self.qnorm(q))
        return [self.outs[j](q[:, j]) for j in range(3)]


ones = lambda x: torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)   # noqa: E731


def student_groups(D, rec):
    return [("q_hist", D["q_hist"], D["q_hist_m"]), ("q_rec", rec, ones(rec)),
            ("t_hist", D["t_hist"], D["t_hist_m"]), ("t_rec", D["t_rec"], ones(D["t_rec"]))]


def teacher_groups(D):
    return [("t_hist", D["t_hist"], D["t_hist_m"]), ("t_rec", D["t_rec"], ones(D["t_rec"])), ("t_fut", D["t_fut"], ones(D["t_fut"]))]


def focal(lg, y):
    lp = F.log_softmax(lg.float(), -1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - lp.exp()) ** a.gamma * lp).mean()


def kd_loss(s, t):
    return sum(F.kl_div(F.log_softmax(s[j].float() / a.T, -1), F.softmax(t[j].float() / a.T, -1), reduction="batchmean")
               for j in range(3)) * a.T ** 2


def cm_recall(lg, y):
    hit = (lg.topk(5, -1).indices == y[:, None]).any(-1)
    per = collections.defaultdict(list)
    for c, h in zip(y.tolist(), hit.tolist()):
        per[c].append(h)
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def scores(lg, y):
    return {"action": cm_recall(lg[0], y[:, 0]), "verb": cm_recall(lg[1], y[:, 1]), "noun": cm_recall(lg[2], y[:, 2]),
            "action_top1": 100 * (lg[0].argmax(-1) == y[:, 0]).float().mean().item()}


# ------------------------------------------------------------------ 1. teacher (TSM oracle) -> logits for all train segments
t0 = time.time()
teacher = Probe().to(dev)
opt = torch.optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=0.05)
N = len(IDX["train"]); EP_T = 20
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP_T * math.ceil(N / 128), pct_start=0.1)
for ep in range(EP_T):
    teacher.train()
    for b in torch.randperm(N).split(128):
        D = cached("train", b, vlm=False)
        lg = teacher(teacher_groups(D))
        loss = sum(focal(lg[j], D["y"][:, j]) for j in range(3))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
teacher.eval()
TL = [torch.zeros(N, n, dtype=torch.float16) for n in SIZES]
with torch.no_grad():
    for b in torch.arange(N).split(512):
        for j, o in enumerate(teacher(teacher_groups(cached("train", b, vlm=False)))):
            TL[j][b] = o.half().cpu()
    vl = [[], [], []]
    for b in torch.arange(len(IDX["validation"])).split(512):
        for j, o in enumerate(teacher(teacher_groups(cached("validation", b, vlm=False)))):
            vl[j].append(o.float())
r = scores([torch.cat(o) for o in vl], Q["validation"]["y"].to(dev))
say(f"teacher (TSM oracle) val: verb {r['verb']:.1f} object {r['noun']:.1f} action {r['action']:.1f} | {(time.time() - t0) / 60:.1f} min")
del teacher, opt

# ------------------------------------------------------------------ 2. Qwen3-VL, cut at the chosen layer, + LoRA
proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
qwen = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                       device_map="cuda")
lm = qwen.model.language_model
lm.layers = nn.ModuleList(list(lm.layers)[: a.layer])
qwen.config.text_config.num_hidden_layers = a.layer
IMG, MERGE = qwen.config.image_token_id, qwen.config.vision_config.spatial_merge_size
grab = {}


def hook(_, __, out):
    grab["h"] = out[0] if isinstance(out, tuple) else out


lm.layers[-1].register_forward_hook(hook)
for p in qwen.parameters():
    p.requires_grad_(False)
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                 target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
peft = get_peft_model(qwen, cfg)
lora_params = [p for p in qwen.parameters() if p.requires_grad]
for p in lora_params:
    p.data = p.data.float()
qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
qwen.enable_input_require_grads()
say(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.1f} M | language layers kept: {len(lm.layers)}")

# warm start: the student probe on CACHED tokens (recent frames from 05_features_grid.py) + KD, as in 04_probe.py
probe = Probe().to(dev)
if a.warm:
    t0 = time.time()
    wopt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
    wsched = torch.optim.lr_scheduler.OneCycleLR(wopt, 3e-4, total_steps=a.warm * math.ceil(N / 128), pct_start=0.1)
    rec_c = lambda s, b: ((Q[s]["rec"][b].float() - MQ) / SQ).reshape(len(b), 4 * C, -1).to(dev)   # noqa: E731
    for ep in range(a.warm):
        probe.train()
        for b in torch.randperm(N).split(128):
            D = cached("train", b)
            lg = probe(student_groups(D, rec_c("train", b)))
            loss = sum(focal(lg[j], D["y"][:, j]) for j in range(3)) + a.lam_kd * kd_loss(lg, [TL[j][b].to(dev) for j in range(3)])
            wopt.zero_grad(); loss.backward(); wopt.step(); wsched.step()
    probe.eval(); vl = [[], [], []]
    with torch.no_grad():
        for b in torch.arange(len(IDX["validation"])).split(512):
            for j, o in enumerate(probe(student_groups(cached("validation", b), rec_c("validation", b)))):
                vl[j].append(o.float())
    r = scores([torch.cat(o) for o in vl], Q["validation"]["y"].to(dev))
    say(f"warm start (cached features, no LoRA) val: verb {r['verb']:.1f} object {r['noun']:.1f} action {r['action']:.1f} "
        f"top-1 {r['action_top1']:.1f} | {(time.time() - t0) / 60:.1f} min")
    del wopt
opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr_lora}, {"params": probe.parameters(), "lr": a.lr_probe}],
                        weight_decay=0.05)


# ------------------------------------------------------------------ data: images -> processor inputs (in workers)
class DS(torch.utils.data.Dataset):
    def __init__(self, split, limit=0):
        self.split, self.rows = split, IDX[split][: limit or None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q = self.rows[i]
        ims = [Image.open(os.path.join(a.vlm, "frames", q["seq"], f"{f:06d}.jpg")).convert("RGB") for f in q["recent"]]
        content = [{"type": "text", "text": "The last 2 seconds, oldest first:"}] + [{"type": "image", "image": im} for im in ims]
        content.append({"type": "text", "text": "The person is assembling or disassembling a toy vehicle. What will they do next?"})
        return i, [{"role": "user", "content": content}]


def collate(batch):
    idx = torch.tensor([b[0] for b in batch])
    inp = proc.apply_chat_template([b[1] for b in batch], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", padding=True)
    return idx, dict(inp)


def runs(pos):
    out, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            out.append((pos[start], pos[i - 1] + 1)); start = i
    return out


def vlm_tokens(inp):
    """live layer-L image tokens of the 4 recent frames -> [B, 4*G*G, 4096], standardized like the cached ones."""
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwen.model(**inp, use_cache=False)
    H = grab.pop("h")
    thw = inp["image_grid_thw"].tolist()
    out, im = [], 0
    for b in range(len(H)):
        rr = runs((inp["input_ids"][b] == IMG).nonzero().flatten().tolist())
        assert len(rr) == 4, len(rr)
        cells = []
        for i0, i1 in rr:
            t, h, w = thw[im]; im += 1
            h, w = h // MERGE, w // MERGE
            x = H[b, i0:i1].float().reshape(t * h, w, -1).permute(2, 0, 1)
            cells.append(F.adaptive_avg_pool2d(x, a.grid).flatten(1).t())     # [G*G, 4096]
        out.append(torch.cat(cells))
    return (torch.stack(out) - MQg) / SQg


def loader(split, shuffle, limit=0):
    return torch.utils.data.DataLoader(DS(split, limit), batch_size=a.bs, shuffle=shuffle, num_workers=a.workers,
                                       collate_fn=collate, persistent_workers=False, prefetch_factor=2)


@torch.no_grad()
def evaluate(split="validation", limit=0):
    qwen.eval(); probe.eval()
    outs, ys = [[], [], []], []
    for idx, inp in loader(split, False, limit):
        D = cached(split, idx)
        for j, o in enumerate(probe(student_groups(D, vlm_tokens(inp)))):
            outs[j].append(o.float())
        ys.append(D["y"])
    return scores([torch.cat(o) for o in outs], torch.cat(ys))


# ------------------------------------------------------------------ 3. train
steps_per_epoch = math.ceil(N / (a.bs * a.accum))
total = a.max_steps or a.epochs * steps_per_epoch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr_lora, a.lr_probe], total_steps=total, pct_start=0.05)
say(f"{steps_per_epoch} optimizer steps per epoch (batch {a.bs} x accum {a.accum}) | {total} steps planned")
step, seen, t1, res = 0, 0, time.time(), []
for ep in range(a.epochs):
    qwen.train(); probe.train()
    for k, (idx, inp) in enumerate(loader("train", True)):
        D = cached("train", idx)
        lg = probe(student_groups(D, vlm_tokens(inp)))
        t = [TL[j][idx].to(dev) for j in range(3)]
        loss = sum(focal(lg[j], D["y"][:, j]) for j in range(3)) + a.lam_kd * kd_loss(lg, t)
        (loss / a.accum).backward()
        seen += len(idx)
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(probe.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t1
                say(f"epoch {ep + 1} step {step}/{total} | loss {loss.item():.3f} | {seen / el:.2f} segments/s | "
                    f"~{(total - step) * a.bs * a.accum / (seen / el) / 3600:.1f} h left | GPU {torch.cuda.max_memory_allocated() / 2**30:.0f} GB")
            if a.max_steps and step >= a.max_steps:
                break
    if a.max_steps:
        r = evaluate(limit=a.val_limit or 200)
        say(f"speed test done: val (first {a.val_limit or 200} segments, not comparable) {r}")
        break
    r = evaluate(limit=a.val_limit); r["epoch"] = ep + 1; res.append(r)
    say(f"epoch {ep + 1} val: verb {r['verb']:.1f} object {r['noun']:.1f} action {r['action']:.1f} top-1 {r['action_top1']:.1f}")
    peft.save_pretrained(os.path.join(a.out, f"lora_ep{ep + 1}"))
    torch.save(probe.state_dict(), os.path.join(a.out, f"probe_ep{ep + 1}.pt"))
    json.dump({"args": vars(a), "results": res}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
say("done")
