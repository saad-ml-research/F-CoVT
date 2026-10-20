#!/bin/bash
# Day 2 prep: download Future-L1-50K and report what it needs (video paths, frame indices).
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
mkdir -p /workspace/data /workspace/F-CoVT/logs
huggingface-cli download Eurayka/Future-L1-50K --repo-type dataset --local-dir /workspace/data/Future-L1-50K
python /workspace/F-CoVT/runs/inspect_data.py /workspace/data/Future-L1-50K 2>&1 | tee /workspace/F-CoVT/logs/02_fl1_data.log
