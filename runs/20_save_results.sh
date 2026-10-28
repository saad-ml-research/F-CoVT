#!/bin/bash
# Save every row's evidence into the repo before stopping the pod (small files only):
#   logs/results/<eval>/samples.jsonl.gz + results.json   (per-question answers -> rescoring / paired tests later)
#   logs/train/<run>/trainer_state.json (+ fcovt_emb_init.txt, fcovt_step.txt)   (training curves)
#   logs/*.log                                            (console logs already written there)
# Adapters/checkpoints stay on the /workspace volume (they survive a pod stop).
set -e
O=/workspace/outputs; R=/workspace/F-CoVT/logs
mkdir -p $R/results $R/train
for d in $O/eval_*; do
  n=$(basename $d); mkdir -p $R/results/$n
  s=$(find $d -name "*samples*.jsonl" | head -1); a=$(find $d -name "*results*.json" | head -1)
  [ -n "$s" ] && gzip -c "$s" > $R/results/$n/samples.jsonl.gz
  [ -n "$a" ] && cp "$a" $R/results/$n/results.json
done
for d in $O/row*; do
  [ -f $d/trainer_state.json ] || continue
  n=$(basename $d); mkdir -p $R/train/$n
  cp $d/trainer_state.json $R/train/$n/
  for f in fcovt_emb_init.txt fcovt_step.txt; do [ -f $d/$f ] && cp $d/$f $R/train/$n/; done
done
du -sh $R/results $R/train
cd /workspace/F-CoVT
git add logs runs
git commit -qm "Results and training curves: rows A-F, ablations, high-res evals" && git push && git log --oneline -1
