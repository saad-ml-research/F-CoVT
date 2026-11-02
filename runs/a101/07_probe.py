"""Linear probe on frozen features: refit the action head on the trained model's hidden states.
The head trained jointly sees each question once (1 epoch, ~10 examples per action) and stays underfit (act_loss
above the 11.5 frequency-only floor). Here the trained model is frozen, every question's hidden state at the end of
"<answer>" is saved once, and a fresh head (same ActHead) is trained on those vectors for many epochs.
Same protocol for every row (B, D, F), so the comparison stays fair.

  extract: python 07_probe.py extract RUN_DIR DATA.json FEATS.pt [--think none|latent|generate] [--blocks 2] [--limit N]
           (prompt / reply built exactly as 06_eval_head.py; saves fp16 hidden states + labels; resumes if interrupted)
  fit:     python 07_probe.py fit TRAIN_FEATS.pt VAL_FEATS.pt OUT_DIR [--epochs 60]
           (10% of train held out to pick the epoch; validation is only scored, never used for choosing)
"""
import argparse, collections, json, math, os, random, re, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "8")   # 128-core pod: dozens of threads fighting over tiny CPU ops
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import torch.nn.functional as F

os.environ.setdefault("FCOVT_STAGED_BLOCKS", "2")   # Assembly101 row F: next moment + step's result
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fcovt"))
import fcovt_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["extract", "fit"])
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("c")
ap.add_argument("--think", default="none", choices=["none", "latent", "generate", "fcovt"],
                help="fcovt = row F's full stage-3 thought template, forced (thought tokens are fixed ids, so no generation)")
ap.add_argument("--feats", default="answer",
                help="fit: answer | thoughts | both, or a comma list of saved keys (h, t1, t2), e.g. h,t1")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--context", type=int, default=4)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--arch", default="linear", choices=["linear", "mlp"], help="fit: head type (mlp = 1 hidden layer)")
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--joint", default="", help="fit: RUN_DIR/act_head.pt, also score the head trained jointly (1 epoch)")
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def extract(run, data_path, out):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    MAXP, MINP = 256 * 28 * 28, 64 * 28 * 28
    proc = AutoProcessor.from_pretrained(os.path.join(run, "merged"))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(run, "merged"), torch_dtype=torch.bfloat16,
                                                               attn_implementation="flash_attention_2", device_map="cuda").eval()
    data = json.load(open(data_path))[: a.limit or None]
    done = torch.load(out) if os.path.exists(out) else {"ids": [], "h": [], "t1": [], "t2": [], "y": [], "think": a.think}
    fcovt_resp = fcovt_core.staged_texts({"stage": 3, "experts": [0, 1, 2]}, "")[2]
    fcovt_resp = fcovt_resp[: fcovt_resp.index("<answer>") + len("<answer>")]   # training cut the reply here (FCOVT_HEAD)
    _ids = [proc.tokenizer.convert_tokens_to_ids(t) for t in (fcovt_core.SEG, fcovt_core.DEP, fcovt_core.DINO, fcovt_core.LAT)]
    unk = proc.tokenizer.unk_token_id
    tid = torch.tensor([i for i in _ids if i is not None and i != unk] or [-1], device="cuda")   # B: no thought tokens
    if a.think == "fcovt":
        assert len(tid) == 4, "this model has no thought tokens"
        print("forced reply:", repr(fcovt_resp[:160]), "...", flush=True)
    seen = set(done["ids"])
    latent = "".join(fcovt_core.block("generic") for _ in range(a.blocks))

    def img(d, k):
        return {"type": "image", "image": Image.open(os.path.join(d, f"f{k}.jpg")).convert("RGB"),
                "max_pixels": MAXP, "min_pixels": MINP}

    def prompt(q):
        cur = q["image"][-1]
        prior = list(range(max(1, cur - a.context), cur))
        content = []
        if prior:
            content.append({"type": "text", "text": "Earlier frames (oldest first):"})
            content += [img(q["video"], k) for k in prior]
            content.append({"type": "text", "text": "Current frame:"})
        for part in re.split(r"(<image>)", q["conversations"][0]["value"]):
            part = part.strip()
            if part == "<image>":
                content.append(img(q["video"], cur))
            elif part:
                content.append({"type": "text", "text": part})
        msgs = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), process_vision_info(msgs)[0]

    todo = [q for q in data if q["id"] not in seen]
    print(f"{len(seen)} already done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        text, ims = prompt(q)
        if a.think == "latent":
            text += latent + "<answer>"
        elif a.think == "fcovt":
            text += fcovt_resp
        elif a.think == "none":
            text += "<answer>"
        else:
            inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=320, do_sample=False, stop_strings=["<answer>"], tokenizer=proc.tokenizer)
            gen = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
            text += gen.split("<answer>")[0].replace("<|im_end|>", "") + "<answer>"
        inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hid = model(**inp, output_hidden_states=True).hidden_states[-1][0]
        h = hid[-1]
        pos = torch.isin(inp["input_ids"][0], tid).nonzero().flatten()       # thought slots, block 1 then block 2
        if len(pos) >= 2:
            half = len(pos) // 2
            done["t1"].append(hid[pos[:half]].mean(0).float().cpu().half())
            done["t2"].append(hid[pos[half:]].mean(0).float().cpu().half())
        done["ids"].append(q["id"]); done["h"].append(h.float().cpu().half())
        done["y"].append([q["action_id"], q["verb_id"], q["noun_id"]])
        if i % 25 == 0 or i == len(todo):
            sp = (time.time() - t0) / i
            print(f"{len(done['ids'])}/{len(data)}  {sp:.2f} s/question  ~{sp*(len(todo)-i)/60:.0f} min left", flush=True)
        if i % 250 == 0 or i == len(todo):
            torch.save(done, out)


class MLPHead(torch.nn.Module):
    """Same outputs as fcovt_core.ActHead, with one hidden layer in between."""
    def __init__(self, d, hidden, drop):
        super().__init__()
        nn = torch.nn
        self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop))
        self.outs = nn.ModuleList(nn.Linear(hidden, n) for n in fcovt_core.HEAD_SIZES)

    def forward(self, h):
        z = self.body(h)
        return tuple(o(z) for o in self.outs)


def metrics(logits3, y):
    out = {}
    for j, kind in enumerate(("action", "verb", "noun")):
        top5 = logits3[j].topk(5, -1).indices
        yy = y[:, j]
        hit1 = (top5[:, 0] == yy); hit5 = (top5 == yy[:, None]).any(-1)
        per = collections.defaultdict(list)
        for c, h in zip(yy.tolist(), hit5.tolist()):
            per[c].append(h)
        out[kind] = {"top1": 100 * hit1.float().mean().item(), "top5": 100 * hit5.float().mean().item(),
                     "class_mean_top5_recall": 100 * sum(sum(v) / len(v) for v in per.values()) / len(per),
                     "classes": len(per)}
    return out


def fit(tr_path, va_path, out_dir):
    torch.manual_seed(0); random.seed(0)
    tr, va = torch.load(tr_path), torch.load(va_path)
    def feats(d):
        parts = {"answer": ["h"], "thoughts": ["t1", "t2"], "both": ["h", "t1", "t2"]}.get(a.feats, a.feats.split(","))
        for k in parts:
            assert len(d.get(k, [])) == len(d["h"]), f"no '{k}' features saved (extracted without thought slots?)"
        return torch.cat([torch.stack(d[k]).float() for k in parts], 1)
    X = feats(tr).to(DEV); Y = torch.tensor(tr["y"]).to(DEV)
    Xv = feats(va).to(DEV); Yv = torch.tensor(va["y"]).to(DEV)
    print(f"features: {a.feats}  dim {X.shape[1]}  train {len(X)}  val {len(Xv)}", flush=True)
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(DEV)
    nh = len(X) // 10
    ho, fi = idx[:nh], idx[nh:]
    head = (MLPHead(X.shape[1], a.hidden, a.drop) if a.arch == "mlp" else fcovt_core.ActHead(X.shape[1])).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def loss_of(lg, y):
        return sum(F.cross_entropy(lg[j], y[:, j]) for j in range(3))

    best, best_state, best_ep = math.inf, None, -1
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = fi[torch.randperm(len(fi), device=fi.device)]
        tot = 0.0
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            l = loss_of(head(X[b]), Y[b])
            opt.zero_grad(); l.backward(); opt.step(); tot += l.item() * len(b)
        sched.step()
        head.eval()
        with torch.no_grad():
            hl = loss_of(head(X[ho]), Y[ho]).item()
        if hl < best:
            best, best_ep = hl, ep
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train loss {tot/len(fi):6.2f}  held-out loss {hl:6.2f}", flush=True)
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        m = metrics(head(Xv), Yv)
        vloss = loss_of(head(Xv), Yv).item()
    pri = {}
    for j, (kind, n) in enumerate(zip(("action", "verb", "noun"), fcovt_core.HEAD_SIZES)):
        c = torch.bincount(Y[:, j], minlength=n).float()
        pri[kind] = c.expand(len(Yv), -1).to(DEV)
    pm = metrics([pri["action"], pri["verb"], pri["noun"]], Yv)
    rows = {"frequency-only": pm}
    if a.joint and a.feats in ("answer", "h"):
        jh = fcovt_core.ActHead(X.shape[1])
        jh.load_state_dict({k: v.float() for k, v in torch.load(a.joint, map_location="cpu").items()})
        jh = jh.to(DEV).eval()
        with torch.no_grad():
            rows["joint head (1 epoch)"] = metrics(jh(Xv), Yv)
    rows[f"probe ({a.arch}, {a.feats})"] = m
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, f"probe_head_{a.arch}_{a.feats}.pt"))
    json.dump({"rows": rows, "best_epoch": best_ep, "heldout_loss": best, "val_loss": vloss,
               "think": tr.get("think"), "arch": a.arch, "hidden": a.hidden, "drop": a.drop}, open(os.path.join(out_dir, f"probe_summary_{a.arch}_{a.feats}.json"), "w"), indent=1)
    print(f"\nbest epoch {best_ep}  held-out loss {best:.2f}  validation loss {vloss:.2f}  (frequency-only floor ~11.5)")
    print(f"{'validation':22s}" + "".join(f"{k + ' top1/top5/cm-rec5':>28s}" for k in ("action", "verb", "noun")))
    for name, r in rows.items():
        print(f"{name:22s}" + "".join(f"{r[k]['top1']:14.1f}{r[k]['top5']:7.1f}{r[k]['class_mean_top5_recall']:7.1f}"
                                      for k in ("action", "verb", "noun")))


if a.cmd == "extract":
    extract(a.a, a.b, a.c)
else:
    fit(a.a, a.b, a.c)
