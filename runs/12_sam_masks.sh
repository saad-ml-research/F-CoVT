#!/bin/bash
# Precompute SAM masks for F-CoVT (row E). Usage: bash runs/12_sam_masks.sh   |   LIMIT=50 bash runs/12_sam_masks.sh (speed test)
set -e
CK=/workspace/CoVT/train/src/anchors/segment_anything/ckpt
mkdir -p $CK /workspace/F-CoVT/logs
[ -f $CK/sam_vit_h_4b8939.pth ] || wget -q --show-progress -O $CK/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
source /workspace/envs/covt/bin/activate
python /workspace/F-CoVT/runs/12_sam_masks.py --limit ${LIMIT:-0} 2>&1 | tee -a /workspace/F-CoVT/logs/12_sam_masks.log
