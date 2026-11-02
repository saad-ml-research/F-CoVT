"""Frozen DINOv2 ViT-L/14 features of the CURRENT frame (f5) and of the TRUE future frames (f6, f7), saved in the
07_probe.py feature format so the same linear probe can be fit on them:
  h = f5 (what every row sees),  t1 = f6 (true next moment),  t2 = f7 (true step result).
Answers: how much does knowing the real future help predict the next action? If even the TRUE f6 barely beats f5,
imagining the future cannot help; if it helps a lot, the gap is what F's thoughts should be recovering.
Feature = [CLS ; mean patch token] (2 x 1024).
usage: python 08_dino_feats.py /workspace/data/a101/validation_head.json /workspace/outputs/dino/val.pt
then:  python 07_probe.py fit TRAIN.pt VAL.pt OUT --feats h | t1 | t2 | h,t1 | h,t1,t2
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TORCH_HOME", "/workspace/torch")
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(8)
data_path, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
data = json.load(open(data_path))
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def load(d, k):
    im = Image.open(os.path.join(d, f"f{k}.jpg")).convert("RGB").resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1)


@torch.no_grad()
def feats(batch):
    x = torch.stack(batch).cuda().float().div(255)
    x = ((x - mean) / std).half()
    o = model.forward_features(x)
    return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1).float().cpu().half()


res = {"ids": [], "h": [], "t1": [], "t2": [], "y": [], "think": "dino_f5_f6_f7"}
t0, B = time.time(), 64
for s in range(0, len(data), B):
    qs = data[s:s + B]
    cur = [q["image"][-1] for q in qs]
    fut = [q["reasoning_image"] for q in qs]
    for key, ks in (("h", cur), ("t1", [f[0] for f in fut]), ("t2", [f[1] for f in fut])):
        res[key] += list(feats([load(q["video"], k) for q, k in zip(qs, ks)]))
    res["ids"] += [q["id"] for q in qs]
    res["y"] += [[q["action_id"], q["verb_id"], q["noun_id"]] for q in qs]
    if (s // B) % 20 == 0:
        print(f"{len(res['ids'])}/{len(data)}  {time.time()-t0:.0f}s", flush=True)
torch.save(res, out)
print("saved", out, len(res["ids"]))
