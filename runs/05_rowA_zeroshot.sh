#!/bin/bash
# Row A: Qwen2.5-VL-7B zero-shot on FutureBench (32 frames), two prompts:
#   futurebench           = direct letter answer
#   futurebench_future_l1 = answer inside <answer></answer> (same prompt we use for B and C)
# Needs FB_DATA (load_from_disk folder) and FB_VIDEOS (V1-33K video root).
set -e
FB_DATA=${FB_DATA:-/workspace/data/futurebench/hf}; FB_VIDEOS=${FB_VIDEOS:-/workspace/data/futurebench/videos}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
TAG=${TAG:-rowA}
TASKS=${TASKS:-"futurebench futurebench_future_l1"}
# Video resolution: the qwen2_5_vl wrapper sizes video frames by a TOTAL pixel budget (total_pixels, default
# 1605632 over 32 frames -> ~210k px/frame); max_pixels only affects still images. qwen-vl-utils caps a video
# frame at 768*28*28 = 602,112 px. TOTALPIX=19267584 (= 32/2 * 602112 * 2) gives every frame that cap (~3x).
TOTALPIX=${TOTALPIX:-1605632}
[ "$TOTALPIX" != "1605632" ] && TAG="${TAG}_tp${TOTALPIX}"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT" && TAG="${TAG}_limit${LIMIT}"
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
T=/workspace/Future-L1/lmms-eval/lmms_eval/tasks/futurebench
for y in futurebench.yaml futurebench_future_l1.yaml futurebench_thinking.yaml; do
  sed -i -e "s#^dataset_path:.*#dataset_path: ${FB_DATA}#" -e "s#^  cache_dir:.*#  cache_dir: ${FB_VIDEOS}/#" $T/$y
done
cd /workspace/Future-L1/lmms-eval
for task in $TASKS; do
  python -m lmms_eval --model qwen2_5_vl \
    --model_args pretrained=${MODEL},use_flash_attention_2=True,max_num_frames=32,max_pixels=$((256*28*28)),total_pixels=${TOTALPIX} \
    --tasks $task --batch_size 1 --log_samples $LIMIT_ARG \
    --output_path /workspace/outputs/eval_${TAG}_${task} 2>&1 | tee /workspace/F-CoVT/logs/05_${TAG}_${task}.log
done
