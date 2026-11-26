"""Average several TempAgg score files (patch_save_scores.py) into one ensemble file: mean of log-softmax scores,
segments matched by id. usage: python avg_npz.py OUT.npz a.npz b.npz ..."""
import sys
import numpy as np

out, paths = sys.argv[1], sys.argv[2:]
zs = [np.load(p) for p in paths]
ids = zs[0]["ids"]
for z in zs[1:]:
    assert np.array_equal(z["ids"], ids), "segment order differs"


def lsm(x):
    x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))


res = {k: np.mean([lsm(z[k].astype(np.float64)) for z in zs], 0) for k in ("verb", "object", "action")}
for k in ("verb_labels", "object_labels", "action_labels", "ids"):
    res[k] = zs[0][k]
np.savez(out, **res)
print(f"TempAgg ensemble of {len(zs)} models -> {out}")
