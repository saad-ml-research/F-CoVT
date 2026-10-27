"""Why do all thought-token runs (D, E g=1, E g=0.1) land at ~61 while B' gets 67?
Two checks, no GPU needed.
 1) Leak check: the number of thought blocks per training sample = min(#future frames, 2). If the answer
    depends on that count, the model can learn a shortcut from the block count during training that is
    wrong at test time (where STEPS is fixed).
 2) Training CE: answer-token CE over the run (B' logs it as `loss`; D/E log it as `train/ce_loss`).
usage: python 18_diag.py /workspace/data/fl1_10k/train.json Bp=/workspace/outputs/rowBctx4_1026_1618 D=... E01=..."""
import json, sys
from collections import Counter, defaultdict

data = json.load(open(sys.argv[1]))
groups = defaultdict(Counter)
for r in data:
    nb = max(1, min(len(r.get("reasoning_image") or []), 2))
    groups[nb][(r.get("answer") or "").strip()[:40]] += 1
print("== 1) answers by number of thought blocks")
for nb in sorted(groups):
    c = groups[nb]; n = sum(c.values())
    print(f"  {nb} block(s): n={n}  distinct answers={len(c)}  top: " + ", ".join(f"{a!r} {100*k/n:.0f}%" for a, k in c.most_common(5)))

print("\n== 2) training CE on answer tokens (mean per quarter of the run)")
for arg in sys.argv[2:]:
    name, run = arg.split("=", 1)
    logs = json.load(open(f"{run}/trainer_state.json"))["log_history"]
    key = "train/ce_loss" if any("train/ce_loss" in l for l in logs) else "loss"
    v = [l[key] for l in logs if key in l]
    q = max(1, len(v) // 4)
    quarters = [sum(v[i:i + q]) / len(v[i:i + q]) for i in range(0, q * 4, q) if v[i:i + q]]
    print(f"  {name:5s} ({key}): " + "  ".join(f"{x:.3f}" for x in quarters))
