"""DINOv2 feature bank for the frame-split study: for every question, frozen DINOv2 ViT-L/14 features of
  past:   every 0.25 s from 8 s before the cut to the cut (33 frames)
  future: +0.5, +1.0 (next action starts), +1.5, +2.0, +3.0 s after the cut, the END of the next action, and the
          end of the coarse step (= f7)
so any past/future split can then be studied in minutes without touching video again (10_split_study.py).
Videos are streamed one recording at a time (next one downloads while the current one is processed) and deleted.
Frames are decoded straight at 224x224 (squashed, as 08_dino_feats.py). Feature = [CLS ; mean patch] (2048).
Resumable: one cache file per recording.
usage: python 09_dino_bank.py /workspace/outputs/dino_bank train validation
"""
import csv, json, os, queue, shutil, sys, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TORCH_HOME", "/workspace/torch")
import torch
import decord
from huggingface_hub import hf_hub_download

torch.set_num_threads(8)
REPO, VIEW = "cvml-nus/assembly101", "C10119_rgb.mp4"
DATA = os.environ.get("A101_DATA", "/workspace/data/a101")
ANN = os.environ.get("A101_ANN", "/workspace/data/assembly101/annotations")
TMP = os.environ.get("A101_TMP", "/workspace/data/a101/_tmp_bank")
PAST = [round(-8 + 0.25 * i, 2) for i in range(33)]
FUT = [0.5, 1.0, 1.5, 2.0, 3.0]
NAMES = [f"{t:+.2f}s" for t in PAST + FUT] + ["next_end", "step_end"]

out_dir, splits = sys.argv[1], sys.argv[2:]
cache = os.path.join(out_dir, "cache")
os.makedirs(cache, exist_ok=True)

# end frame of the next action, looked up by (recording, start frame) -- questions only keep its start
ends = {}
for sp in splits:
    with open(os.path.join(ANN, "fine-grained-annotations", f"{sp}.csv")) as f:
        for r in csv.DictReader(f):
            if r["video"].endswith("/" + VIEW):
                ends[(r["video"].split("/")[0], int(r["start_frame"]))] = int(r["end_frame"])

qs = {sp: json.load(open(os.path.join(DATA, f"{sp}_head.json"))) for sp in splits}
by_seq = {}
for sp, lst in qs.items():
    for q in lst:
        by_seq.setdefault(q["seq"], []).append(q)
seqs = sorted(by_seq)


def frames30(q):
    cut = q["cut_frame"]
    return [cut + round(t * 30) for t in PAST + FUT] + [ends[(q["seq"], q["next_start"])], q["frames_30fps"][6]]


todo = [s for s in seqs if not os.path.exists(os.path.join(cache, f"{s}.pt"))]
print(f"{len(seqs)} recordings, {sum(len(v) for v in by_seq.values())} questions, {len(todo)} recordings to do, "
      f"{len(NAMES)} frames per question", flush=True)

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


@torch.no_grad()
def embed(arr):                                   # uint8 [B, 224, 224, 3] -> [B, 2048] fp16 (cpu)
    x = torch.from_numpy(arr).cuda().permute(0, 3, 1, 2).float().div(255)
    o = model.forward_features(((x - mean) / std).half())
    return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1).cpu()


ready = queue.Queue(maxsize=2)                    # download the next recordings while this one is processed


def downloader():
    for s in todo:
        d = os.path.join(TMP, s)
        for attempt in range(3):
            try:
                p = hf_hub_download(REPO, f"recordings/{s}/{VIEW}", repo_type="dataset", local_dir=d)
                break
            except Exception as e:                # transient network errors: retry, then give up on this one
                print(f"download {s} failed ({e}), retry {attempt + 1}", flush=True); p = None; time.sleep(10)
        ready.put((s, p))
    ready.put(None)


threading.Thread(target=downloader, daemon=True).start()
t0, done = time.time(), 0
while True:
    item = ready.get()
    if item is None:
        break
    s, path = item
    if path is None:
        continue
    vr = decord.VideoReader(path, width=224, height=224, num_threads=8)
    fps, n = vr.get_avg_fps(), len(vr)
    per_q = {q["id"]: frames30(q) for q in by_seq[s]}
    idx = {fr: min(n - 1, max(0, round(fr * fps / 30))) for fl in per_q.values() for fr in fl}
    want = sorted(set(idx.values()))
    feats = {}
    for c in range(0, len(want), 256):
        chunk = want[c:c + 256]
        for i, f in zip(chunk, embed(vr.get_batch(chunk).asnumpy())):
            feats[i] = f
    res = {qid: {"f": torch.stack([feats[idx[fr]] for fr in fl]),
                 "clamped": torch.tensor([fr * fps / 30 < 0 or round(fr * fps / 30) > n - 1 for fr in fl])}
           for qid, fl in per_q.items()}
    torch.save(res, os.path.join(cache, f"{s}.pt"))
    del vr
    shutil.rmtree(os.path.join(TMP, s), ignore_errors=True)
    done += 1
    el = time.time() - t0
    print(f"[{done}/{len(todo)}] {s}: {len(per_q)} questions, {len(want)} frames | {el/60:.1f} min, "
          f"~{el/done*(len(todo)-done)/60:.0f} min left", flush=True)

for sp, lst in qs.items():                        # one file per split, same order as <split>_head.json
    ids, F, C, Y = [], [], [], []
    loaded = {}
    for q in lst:
        if q["seq"] not in loaded:
            p = os.path.join(cache, f"{q['seq']}.pt")
            loaded = {q["seq"]: torch.load(p) if os.path.exists(p) else {}}
        r = loaded[q["seq"]].get(q["id"])
        if r is None:
            continue
        ids.append(q["id"]); F.append(r["f"]); C.append(r["clamped"])
        Y.append([q["action_id"], q["verb_id"], q["noun_id"]])
    torch.save({"ids": ids, "feats": torch.stack(F), "clamped": torch.stack(C), "y": torch.tensor(Y),
                "names": NAMES}, os.path.join(out_dir, f"{sp}.pt"))
    print(f"saved {sp}: {len(ids)}/{len(lst)} questions, feats {tuple(torch.stack(F).shape)}", flush=True)
