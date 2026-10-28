#!/bin/bash
# Eval rows D/E on FutureBench with their thought tokens forced before the answer.
# usage: MODE=fcovt RUN=/workspace/outputs/rowEctx4_xxx [STEPS=2] [LIMIT=20] bash runs/14_eval_thoughts.sh
# row F (CoVT curriculum) writes its own thoughts: MODE=fcovt_staged RUN=... STEPS=0
set -e
: "${MODE:?fcovt or fcovt_generic}"; : "${RUN:?run dir with merged/}"
STEPS=${STEPS:-2}
M=$([ "$MODE" = fcovt ] && echo fcovt || echo generic)
SUF=s$STEPS
if [ -n "$PREFIX" ]; then export FCOVT_PREFIX="$PREFIX"; SUF=${PTAG:-p}   # e.g. PREFIX="<think>" PTAG=think: start the answer, model writes the rest
elif [ "$STEPS" = 0 ]; then export FCOVT_PREFIX=""   # STEPS=0: no thought tokens at all (diagnostic)
else export FCOVT_PREFIX=$(cd /workspace/F-CoVT/runs/fcovt && /workspace/envs/fl1/bin/python -c "import fcovt_core as f; print(f.thought_text('$M', $STEPS), end='')"); fi
echo "prefix: ${FCOVT_PREFIX:0:80}... (${#FCOVT_PREFIX} chars)"
export FB_DATA=${FB_DATA:-/workspace/data/futurebench/hf} FB_VIDEOS=${FB_VIDEOS:-/workspace/data/futurebench/videos}
MODEL=$RUN/merged TAG=$(basename $RUN)_$SUF TASKS=futurebench_future_l1 bash /workspace/F-CoVT/runs/05_rowA_zeroshot.sh
