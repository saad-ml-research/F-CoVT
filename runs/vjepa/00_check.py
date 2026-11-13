"""Check what the released V-JEPA 2 checkpoint gives us before building on it:
  1. are the PREDICTOR weights in the checkpoint (not randomly initialized)?
  2. how does the predictor place target tokens (positions from mask indices into which grid)?
  3. can it predict tokens for time slots AFTER the encoded clip (the future), with the encoder seeing only the past?
usage: python 00_check.py [facebook/vjepa2-vitl-fpc64-256]
"""
import inspect, os, sys
os.environ.setdefault("HF_HOME", "/workspace/hf")
import torch
import transformers
from transformers import AutoModel
from transformers.utils import logging

name = sys.argv[1] if len(sys.argv) > 1 else "facebook/vjepa2-vitl-fpc64-256"
logging.set_verbosity_info()                      # shows "newly initialized" weights, if any
model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
logging.set_verbosity_warning()
cfg = model.config
print("\n=== config:", {k: getattr(cfg, k, None) for k in
      ("frames_per_clip", "crop_size", "patch_size", "tubelet_size", "hidden_size", "pred_hidden_size",
       "pred_num_hidden_layers", "pred_num_mask_tokens", "num_hidden_layers")})
print("=== has predictor:", hasattr(model, "predictor"),
      "| predictor params:", sum(p.numel() for p in model.predictor.parameters()) / 1e6, "M")
print("=== VJEPA2Model.forward signature:", inspect.signature(model.forward))
src = inspect.getsource(type(model).forward)
print("=== VJEPA2Model.forward (first 60 lines):\n" + "\n".join(src.splitlines()[:60]))
for cls in (type(model.predictor), type(model.predictor.embeddings)):
    s = inspect.getsource(cls.forward)
    print(f"=== {cls.__name__}.forward:\n" + "\n".join(s.splitlines()[:45]))
mod = sys.modules[type(model).__module__]
for fn in ("apply_masks",):
    if hasattr(mod, fn):
        print(f"=== {fn}:\n" + inspect.getsource(getattr(mod, fn)))
for n, c in inspect.getmembers(mod, inspect.isclass):
    if "Rope" in n or "RoPE" in n:
        print(f"=== {n}.forward:\n" + "\n".join(inspect.getsource(c.forward).splitlines()[:40]))
print("transformers", transformers.__version__)
