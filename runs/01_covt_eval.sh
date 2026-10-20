#!/bin/bash
# Day 1: reproduce CoVT on CV-Bench. Expect roughly 80.0 (2D) / 74.5 (3D)-ish for seg_depth_dino; compare to base Qwen2.5-VL.
set -e
source /workspace/envs/covt/bin/activate
export HF_HOME=/workspace/hf
cd /workspace/CoVT/VLMEvalKit
mkdir -p /workspace/F-CoVT/logs
python run.py --data CV-Bench-2D CV-Bench-3D \
  --model Qwen2.5-VL-7B-Instruct CoVT-7B-seg_depth_dino \
  --work-dir /workspace/outputs/covt_eval --verbose 2>&1 | tee /workspace/F-CoVT/logs/01_covt_eval.log
# copy the score files into the repo so they can be read from GitHub
find /workspace/outputs/covt_eval -name "*acc*.csv" -o -name "*score*.csv" | xargs -I{} cp {} /workspace/F-CoVT/logs/ 2>/dev/null || true
