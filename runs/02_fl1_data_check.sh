#!/bin/bash
# Download ONLY the Future-L1-50K JSON + README + one parquet shard (2 GB), then inspect them.
# The full repo is ~258 GB (129 x 2 GB parquet shards) - never download it all.
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
D=/workspace/data/Future-L1-50K
mkdir -p $D /workspace/F-CoVT/logs
hf download Eurayka/Future-L1-50K --repo-type dataset --local-dir $D \
  --include "Future-L1-50K.json" "README.md" "data/train-00000-of-00129.parquet"
python /workspace/F-CoVT/runs/inspect_data.py $D 2>&1 | tee /workspace/F-CoVT/logs/02_fl1_data.log
