#!/bin/bash
# Row D: Future-L1-style generic latent, SFT only, on Qwen2.5-VL-7B, 1 x A100, LoRA r16/a32 (CoVT recipe).
# Paper settings: latent spans <= 4, lambda 0.1 (the released script uses 32 / 0.2).
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/Future-L1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
DATA=${DATA:-/workspace/data/fl1_10k.json}
N_STEPS_SMOKE=${SMOKE:-0}            # SMOKE=50 for a quick speed test
OUT=/workspace/outputs/rowD_fl1_qwen25_$(date +%m%d_%H%M)
cd /workspace/Future-L1
EXTRA=""; [ "$N_STEPS_SMOKE" -gt 0 ] && EXTRA="--max_steps $N_STEPS_SMOKE"
torchrun --nproc_per_node 1 src/train/train.py \
  --run_name rowD --deepspeed scripts/zero2.json \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct --data_path $DATA --use_mixed_dataset True \
  --remove_unused_columns False \
  --lora_enable True --lora_rank 16 --lora_alpha 32 --lora_dropout 0.01 \
  --freeze_vision_tower True --freeze_merger True --freeze_llm True \
  --latent_loss mse --latent_lambda 0.1 --max_latent_token 4 \
  --learning_rate 5e-5 --weight_decay 0.1 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --bf16 True --tf32 True --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --num_train_epochs 1 $EXTRA \
  --image_min_pixels $((64*28*28)) --image_max_pixels $((256*28*28)) \
  --video_max_pixels $((128*28*28)) --nframes 16 \
  --lazy_preprocess True --dataloader_num_workers 8 \
  --save_strategy steps --save_steps 200 --save_total_limit 3 --logging_steps 5 \
  --output_dir $OUT --report_to none 2>&1 | tee /workspace/F-CoVT/logs/03_rowD_$(basename $OUT).log
