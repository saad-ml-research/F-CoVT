#!/bin/bash
# Usage: bash runs/07_08_data.sh inspect   (FutureBench format only, seconds)
#        bash runs/07_08_data.sh fb        (FutureBench videos, streams ~156 GB of tars, keeps ~few GB)
#        bash runs/07_08_data.sh train     (10K training samples as frames, ~26 x 2 GB shards, keeps ~5 GB)
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
mkdir -p /workspace/F-CoVT/logs
case "$1" in
  inspect) python /workspace/F-CoVT/runs/08_futurebench_fetch.py --inspect 2>&1 | tee /workspace/F-CoVT/logs/08_fb_inspect.log ;;
  fb)      python /workspace/F-CoVT/runs/08_futurebench_fetch.py 2>&1 | tee /workspace/F-CoVT/logs/08_fb_fetch.log ;;
  train)   python /workspace/F-CoVT/runs/07_fl1_frames.py --shards ${SHARDS:-26} 2>&1 | tee /workspace/F-CoVT/logs/07_fl1_frames.log ;;
  *) echo "usage: $0 inspect|fb|train"; exit 1 ;;
esac
