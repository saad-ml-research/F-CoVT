#!/bin/bash
# One-time setup for rows D/E: expert checkpoints + the F-CoVT patch on Future-L1.
set -e
CK=/workspace/CoVT/train/src/anchors/DepthAnything/ckpt; mkdir -p $CK
[ -f $CK/depth_anything_v2_vitl.pth ] || wget -q --show-progress -O $CK/depth_anything_v2_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
ls -la /workspace/CoVT/train/src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth
cd /workspace/Future-L1
if git apply --check /workspace/F-CoVT/runs/patches/fl1_fcovt.patch 2>/dev/null; then git apply /workspace/F-CoVT/runs/patches/fl1_fcovt.patch && echo "applied fl1_fcovt"
else echo "fl1_fcovt not applied (already applied?)"; fi
if git apply --check /workspace/F-CoVT/runs/patches/fl1_fcovt_staged.patch 2>/dev/null; then git apply /workspace/F-CoVT/runs/patches/fl1_fcovt_staged.patch && echo "applied fl1_fcovt_staged (row F)"
else echo "fl1_fcovt_staged not applied (already applied?)"; fi
if git apply --check /workspace/F-CoVT/runs/patches/fl1_fcovt_staged_fix.patch 2>/dev/null; then git apply /workspace/F-CoVT/runs/patches/fl1_fcovt_staged_fix.patch && echo "applied fl1_fcovt_staged_fix"
else echo "fl1_fcovt_staged_fix not applied (already applied?)"; fi
if git apply --check /workspace/F-CoVT/runs/patches/fl1_fcovt_staged_dbg.patch 2>/dev/null; then git apply /workspace/F-CoVT/runs/patches/fl1_fcovt_staged_dbg.patch && echo "applied fl1_fcovt_staged_dbg"
else echo "fl1_fcovt_staged_dbg not applied (already applied?)"; fi
grep -c "F-CoVT" src/train/train.py src/trainer/future_l1_trainer.py src/dataset/twiff_sft_dataset.py lmms-eval/lmms_eval/models/qwen2_5_vl.py
# DINOv2 via torch.hub into /workspace so it survives restarts
TORCH_HOME=/workspace/torch /workspace/envs/fl1/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14'); print('dinov2 ok')"
