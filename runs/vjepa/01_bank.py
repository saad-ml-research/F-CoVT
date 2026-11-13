"""V-JEPA 2 token bank for model A v2 (Assembly101, camera C10119). Per question (cut = 1 s before the next action):
  past  : frozen V-JEPA 2 ViT-L encoder on 16 frames at 8 fps ending at the cut (-1.875 .. 0 s) -> time slots 0-7
  pred  : the frozen pretrained V-JEPA 2 PREDICTOR, given only the past tokens, predicts slots 8-15 (0 .. +2 s)
          (the encoder never sees a future frame)
  real  : the encoder on 32 frames (-1.875 .. +2.125 s), slots 8-15 = the real future (oracle / JEPA target)
One slot = 2 frames = 0.25 s; the next action starts at slot 11 (+1 s); +1.5 s = slot 13; "t4_slot" = slot of the
end of the next action (clamped to 8..15). Each slot's 16x16 patch grid is average-pooled to 4x4 (16 tokens x 1024).
Frames: short side 256, center crop 256x256, ImageNet normalization (as V-JEPA 2).
Sanity check printed per recording: cosine(pred, real) vs cosine(copy last past slot, real), per future slot.
Resumable (one cache file per recording).
usage: python 01_bank.py /workspace/outputs/vjepa_bank train validation      (A101_LIMIT=2 for a smoke test)
"""
import csv, json, os, queue, shutil, sys, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HF_HOME", "/workspace/hf")
import torch
import torch.nn.functional as F
import decord
from huggingface_hub import hf_hub_download
from transformers import AutoModel

torch.set_num_threads(8)
REPO, VIEW = "cvml-nus/assembly101", "C10119_rgb.mp4"
MODEL = os.environ.get("VJEPA_MODEL", "facebook/vjepa2-vitl-fpc64-256")
DATA = os.environ.get("A101_DATA", "/workspace/data/a101")
ANN = os.environ.get("A101_ANN", "/workspace/data/assembly101/annotations")
TMP = os.environ.get("A101_TMP", "/workspace/data/a101/_tmp_vjepa")
FPS, NPAST, NALL, GRID, POOL = 8, 16, 32, 16, 4
SPP = GRID * GRID                                   # tokens per time slot (16x16)
PS, FS = NPAST // 2, NALL // 2                      # past slots 0..7, all slots 0..15
BQ = int(os.environ.get("VJEPA_BATCH", "8"))

out_dir, splits = sys.argv[1], sys.argv[2:]
cache = os.path.join(out_dir, "cache")
os.makedirs(cache, exist_ok=True)

ends = {}
for sp in splits:
    with open(os.path.join(ANN, "fine-grained-annotations", f"{sp}.csv")) as f:
        for r in csv.DictReader(f):
            if r["video"].endswith("/" + VIEW):
                ends[(r["video"].split("/")[0], int(r["start_frame"]))] = int(r["end_frame"])
qs = {sp: json.load(open(os.path.join(DATA, f"{sp}_head.json"))) for sp in splits}
by_seq = {}
for lst in qs.values():
    for q in lst:
        by_seq.setdefault(q["seq"], []).append(q)
seqs = sorted(by_seq)
todo = [s for s in seqs if not os.path.exists(os.path.join(cache, f"{s}.pt"))]
if os.environ.get("A101_LIMIT"):
    todo = todo[: int(os.environ["A101_LIMIT"])]
print(f"{len(seqs)} recordings, {len(todo)} to do | {MODEL}", flush=True)

model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)
ctx_idx = torch.arange(PS * SPP, device="cuda")                       # past tokens, slots 0-7
tgt_idx = torch.arange(PS * SPP, FS * SPP, device="cuda")             # future tokens, slots 8-15


def times30(q):
    """30-fps annotation frame of each of the 32 clip frames: (k - 15) / 8 s around the cut."""
    return [q["cut_frame"] + round((k - (NPAST - 1)) / FPS * 30) for k in range(NALL)]


def pool(tok, nslots):                                                # [B, nslots*256, D] -> [B, nslots, 16, D]
    B, _, D = tok.shape
    x = tok.reshape(B * nslots, GRID, GRID, D).permute(0, 3, 1, 2).float()
    x = F.adaptive_avg_pool2d(x, POOL)
    return x.permute(0, 2, 3, 1).reshape(B, nslots, POOL * POOL, D).half()


@torch.no_grad()
def encode(clips):                                                    # uint8 [B, T, 256, 256, 3]
    x = torch.from_numpy(clips).cuda().permute(0, 1, 4, 2, 3).float().div(255)
    x = ((x - mean) / std).half()
    return model(pixel_values_videos=x, skip_predictor=True).last_hidden_state


@torch.no_grad()
def predict(past_tokens):
    B = len(past_tokens)
    out = model.predictor(encoder_hidden_states=past_tokens,
                          context_mask=[ctx_idx.unsqueeze(0).repeat(B, 1)],
                          target_mask=[tgt_idx.unsqueeze(0).repeat(B, 1)])
    return out.last_hidden_state                                      # [B, 8*256, D]


ready = queue.Queue(maxsize=2)


def downloader():
    for s in todo:
        p = None
        for attempt in range(3):
            try:
                p = hf_hub_download(REPO, f"recordings/{s}/{VIEW}", repo_type="dataset", local_dir=os.path.join(TMP, s))
                break
            except Exception as e:
                print(f"download {s} failed ({e}), retry {attempt + 1}", flush=True); time.sleep(10)
        ready.put((s, p))
    ready.put(None)


threading.Thread(target=downloader, daemon=True).start()
t0, done, sanity = time.time(), 0, []
while True:
    item = ready.get()
    if item is None:
        break
    s, path = item
    if path is None:
        continue
    vr0 = decord.VideoReader(path)
    h0, w0 = vr0[0].shape[:2]
    del vr0
    nh, nw = (256, round(w0 * 256 / h0)) if h0 <= w0 else (round(h0 * 256 / w0), 256)
    vr = decord.VideoReader(path, width=nw, height=nh, num_threads=8)
    vfps, n = vr.get_avg_fps(), len(vr)
    top, left = (nh - 256) // 2, (nw - 256) // 2
    qlist = by_seq[s]
    res = {}
    for c in range(0, len(qlist), BQ):
        batch = qlist[c:c + BQ]
        idx = [[min(n - 1, max(0, round(fr * vfps / 30))) for fr in times30(q)] for q in batch]
        flat = sorted({i for row in idx for i in row})
        frames = vr.get_batch(flat).asnumpy()[:, top:top + 256, left:left + 256]
        pos = {i: j for j, i in enumerate(flat)}
        clips = frames[[[pos[i] for i in row] for row in idx]]       # [B, 32, 256, 256, 3]
        past_tok = encode(clips[:, :NPAST])                           # [B, 2048, D]
        pred_tok = predict(past_tok)                                  # [B, 2048, D] slots 8-15
        all_tok = encode(clips)                                       # [B, 4096, D]
        real_tok = all_tok[:, PS * SPP:]
        # sanity: predicted vs real, and "copy the last past slot" vs real, per future slot (full 16x16 grid)
        last = past_tok[:, (PS - 1) * SPP:PS * SPP].float()
        for k in range(FS - PS):
            r = real_tok[:, k * SPP:(k + 1) * SPP].float()
            sanity.append((k, F.cosine_similarity(pred_tok[:, k * SPP:(k + 1) * SPP].float(), r, -1).mean().item(),
                           F.cosine_similarity(last, r, -1).mean().item()))
        P, PR, R = pool(past_tok, PS), pool(pred_tok, FS - PS), pool(real_tok, FS - PS)
        for b, q in enumerate(batch):
            e = ends[(q["seq"], q["next_start"])]
            k_end = (NPAST - 1) + round((e - q["cut_frame"]) / 30 * FPS)
            res[q["id"]] = {"past": P[b].cpu(), "pred": PR[b].cpu(), "real": R[b].cpu(),
                            "t4_slot": max(PS, min(FS - 1, k_end // 2))}
    torch.save(res, os.path.join(cache, f"{s}.pt"))
    del vr
    shutil.rmtree(os.path.join(TMP, s), ignore_errors=True)
    done += 1
    el = time.time() - t0
    by_k = {}
    for k, cp, cc in sanity:
        by_k.setdefault(k, [[], []]); by_k[k][0].append(cp); by_k[k][1].append(cc)
    msg = " ".join(f"s{k + PS}:{sum(v[0]) / len(v[0]):.2f}/{sum(v[1]) / len(v[1]):.2f}" for k, v in sorted(by_k.items()))
    print(f"[{done}/{len(todo)}] {s}: {len(qlist)} q | {el / 60:.1f} min, ~{el / done * (len(todo) - done) / 60:.0f} min left"
          f" | cos pred/copy so far {msg}", flush=True)

for sp, lst in qs.items():
    ids, P, PR, R, T4, Y = [], [], [], [], [], []
    loaded = {}
    for q in lst:
        if q["seq"] not in loaded:
            p = os.path.join(cache, f"{q['seq']}.pt")
            loaded = {q["seq"]: torch.load(p) if os.path.exists(p) else {}}
        r = loaded[q["seq"]].get(q["id"])
        if r is None:
            continue
        ids.append(q["id"]); P.append(r["past"]); PR.append(r["pred"]); R.append(r["real"]); T4.append(r["t4_slot"])
        Y.append([q["action_id"], q["verb_id"], q["noun_id"]])
    if not ids:
        print(f"saved {sp}: nothing", flush=True); continue
    torch.save({"ids": ids, "past": torch.stack(P), "pred": torch.stack(PR), "real": torch.stack(R),
                "t4_slot": torch.tensor(T4), "y": torch.tensor(Y), "slot_seconds": 0.25, "first_future_slot": PS},
               os.path.join(out_dir, f"{sp}.pt"))
    print(f"saved {sp}: {len(ids)}/{len(lst)} questions, past {tuple(torch.stack(P).shape)}", flush=True)
