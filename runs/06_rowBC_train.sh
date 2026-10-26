#!/bin/bash
# Rows B and C: LoRA SFT of Qwen2.5-VL-7B on the same 10K training subset.
#   MODE=answer -> row B (question frames -> answer only)
#   MODE=cot    -> row C (question frames -> text reasoning -> answer)
# Future frames are never shown in either mode. SMOKE=50 for a speed test.
set -e
MODE=${MODE:?set MODE=answer or MODE=cot}
DATA=${DATA:-/workspace/data/fl1_10k/train.json}
SMOKE=${SMOKE:-0}
TAG=$([ "$MODE" = answer ] && echo rowB || echo rowC)
CONTEXT=${CONTEXT:-0}                 # CONTEXT=4: also show up to 4 earlier frames (rows B', C')
[ "$CONTEXT" -gt 0 ] && TAG="${TAG}ctx${CONTEXT}"
OUT=/workspace/outputs/${TAG}_$(date +%m%d_%H%M)
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/Future-L1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export FCOVT_RESPONSE_MODE=$MODE FCOVT_CONTEXT=$CONTEXT
cd /workspace/Future-L1
EXTRA=""; [ "$SMOKE" -gt 0 ] && EXTRA="--max_steps $SMOKE"
torchrun --nproc_per_node 1 src/train/train.py \
  --run_name $TAG --deepspeed scripts/zero2.json \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct --data_path $DATA --use_twiff_dataset True \
  --system_message_mode plain --remove_unused_columns False \
  --lora_enable True --lora_rank 16 --lora_alpha 32 --lora_dropout 0.01 \
  --freeze_vision_tower True --freeze_merger True --freeze_llm True \
  --latent_loss mse --latent_lambda 0.0 \
  --learning_rate 5e-5 --weight_decay 0.1 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --bf16 True --tf32 True --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --num_train_epochs 1 $EXTRA \
  --image_min_pixels $((64*28*28)) --image_max_pixels $((256*28*28)) \
  --lazy_preprocess True --dataloader_num_workers 8 --random_seed 42 \
  --save_strategy steps --save_steps 200 --save_total_limit 2 --logging_steps 5 \
  --output_dir $OUT --report_to none 2>&1 | tee /workspace/F-CoVT/logs/06_$(basename $OUT).log
[ "$SMOKE" -gt 0 ] && exit 0
python /workspace/F-CoVT/runs/merge_lora.py $OUT $OUT/merged
echo "eval with: MODEL=$OUT/merged TAG=$TAG bash runs/05_rowA_zeroshot.sh"
