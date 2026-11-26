#!/bin/bash
# Train N more TempAgg models on C10119 (same recipe as the official one; no seed option, so each run gets a different
# random init), score each on validation (--save_scores), then build a TempAgg ensemble (official + retrains) and
# evaluate ours against it.  usage: bash runs/tempagg/train_seeds.sh 4
set -e
N=${1:-4}
TA=/workspace/assembly101-action-anticipation/tempAgg-action-anticipation
F=/workspace/data/a101_feats/TSM_features; A=/workspace/data/a101_anticip/data/CSVs; O=/workspace/outputs
EXP=anti_mod_fixed_view4_span_6_s1_5_s2_3_s3_2_recent_2_r1_1.6_r2_1.2_r3_0.8_r4_0.4_bs_32_drop_0.3_lr_0.0001_dimLa_512_dimLi_512_epoc_15_vb_nn
source /workspace/envs/fl1/bin/activate
cd $TA
for s in $(seq 1 $N); do
  D=$O/tempagg_seed$s
  if [ ! -f $O/tempagg_seed${s}_scores.npz ]; then
    mkdir -p $D
    echo "=== TempAgg retrain $s/$N started $(date -u +%H:%M) UTC"
    python main.py --mode train --path_to_data $F --path_to_anno $A --path_to_models $D --modality fixed --views view4 \
      --batch_size 32 --num_workers 6 2>&1 | grep -E "\[validation\]" | tail -1
    python main.py --mode validate --path_to_data $F --path_to_anno $A --path_to_models $D --model_name_ev ${EXP}_best.pth.tar \
      --modality fixed --views view4 --batch_size 32 --num_workers 6 --save_scores $O/tempagg_seed${s}_scores.npz 2>&1 \
      | grep -E "Top-5 Recall \(Action\)|saved"
    rm -f $D/*.pth.tar                      # 2.2 GB each; the scores are all we need
  fi
done
python /workspace/F-CoVT/runs/tempagg/avg_npz.py $O/tempagg_ensemble_scores.npz $O/tempagg_official_scores.npz $O/tempagg_seed*_scores.npz
python /workspace/F-CoVT/runs/tempagg/fuse_eval.py $O/tempagg_ensemble_scores.npz $O/val_preds /workspace/data/a101_vlm/validation.json $A \
  --rows 'fuse_kd,fuse_kd+hS,fuse_kd+hS|hlab+hS256' --seeds 5
