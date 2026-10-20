#!/bin/bash
# Get FutureBench (1,056 MCQs; videos from V1-33K, NEP paper arXiv 2505.22457).
# Step 1 only LISTS files and sizes so we don't pull huge video dumps by accident.
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
mkdir -p /workspace/F-CoVT/logs
python - <<'PY' 2>&1 | tee /workspace/F-CoVT/logs/04_futurebench_list.log
from huggingface_hub import HfApi
api = HfApi()
for rid in ["haonan3/V1-33K", "haonan3/FutureBench", "Eurayka/Future-L1-Bench", "Eurayka/FutureBench"]:
    try:
        info = api.dataset_info(rid, files_metadata=True)
        files = [(s.rfilename, (s.size or 0) / 1e9) for s in info.siblings]
        print(f"\n== {rid}: {len(files)} files, {sum(f[1] for f in files):.1f} GB total")
        for name, gb in files[:40]:
            print(f"  {gb:7.2f} GB  {name}")
    except Exception as e:
        print(f"\n== {rid}: not found ({type(e).__name__})")
PY
echo "Push logs/04_futurebench_list.log; we pick what to download from it."
