"""V-JEPA 2.1 smoke test: load an encoder from torch hub (facebookresearch/vjepa2), encode random 16-frame 384 px
clips, report the token shape, GPU memory and clips per second, and extrapolate the time for the official
anticipation set (54,723 segments, past 2 s only).
usage: python 00_speed.py vjepa2_1_vit_gigantic_384     (or vjepa2_1_vit_large_384)
"""
import os, sys, time
os.environ.setdefault("TORCH_HOME", "/workspace/torch")
import torch

name = sys.argv[1] if len(sys.argv) > 1 else "vjepa2_1_vit_gigantic_384"
t0 = time.time()
obj = torch.hub.load("facebookresearch/vjepa2", name)
enc = obj[0] if isinstance(obj, (tuple, list)) else obj
print(f"loaded {name} in {time.time() - t0:.0f}s | returned {type(obj).__name__}"
      f"{f' of {len(obj)}' if isinstance(obj, (tuple, list)) else ''} | encoder params {sum(p.numel() for p in enc.parameters()) / 1e9:.2f} B",
      flush=True)
enc = enc.cuda().eval()          # fp32 weights + bf16 autocast: its rotary embeddings run in fp32
N_SEG = 54723
for bs in (1, 4, 8, 16):
    try:
        x = torch.randn(bs, 3, 16, 384, 384, device="cuda")   # [B, C, T, H, W]
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = enc(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(3):
                out = enc(x)
            torch.cuda.synchronize()
        dt = (time.time() - t) / 3
        tok = out[0] if isinstance(out, (tuple, list)) else out
        print(f"batch {bs:2d}: tokens {tuple(tok.shape)} | {bs / dt:5.1f} clips/s | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
              f" | official set (past 2 s): ~{N_SEG / (bs / dt) / 3600:.1f} h of encoder time", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"batch {bs:2d}: out of memory"); break
