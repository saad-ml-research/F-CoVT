#!/bin/bash
# 10-step runs isolating each expert loss, to see which one causes the large gradient norm.
cd /workspace/F-CoVT
run() {  # name, env...
  name=$1; shift
  rm -rf /workspace/outputs/rowEctx4_*
  env "$@" MODE=fcovt CONTEXT=4 SMOKE=10 bash runs/06_rowBC_train.sh > /workspace/gc_$name.txt 2>&1
  echo "== $name: $(grep -o "'grad_norm': [0-9.]*" /workspace/gc_$name.txt | tail -2 | tr '\n' ' ') $(grep -o "'train/[a-z_]*': [0-9.]*" /workspace/gc_$name.txt | tail -4 | tr '\n' ' ')"
}
run none   FCOVT_GAMMA=0
run seg    FCOVT_W_DEPTH=0 FCOVT_W_DINO=0
run depth  FCOVT_W_SEG=0   FCOVT_W_DINO=0
run dino   FCOVT_W_SEG=0   FCOVT_W_DEPTH=0
rm -rf /workspace/outputs/rowEctx4_*
