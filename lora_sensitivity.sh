#!/bin/bash
# 快速LoRA敏感性分析

FILE_PATH="milano.h5"
DEVICE="cuda:3"

echo "=== LoRA Rank 敏感性分析 ==="
for LLM in BERT GPT2; do
  for rank in 4 8 16 32 64; do
    for DATA in call net sms; do
        echo "Testing LoRA rank: $rank"
        EXP_NAME="simpletimellm_milano_${DATA}_perfedavg_${LLM}_rank_${rank}"
        python main.py \
          --model_type simpletimellm \
          --file_path ${FILE_PATH} \
          --experiment_name ${EXP_NAME} \
          --data_type ${DATA} \
          --llm_model ${LLM}\
          --llm_dim 768\
          --seq_len 96 \
          --pred_len 24 \
          --local_ep 5 \
          --epoch 10 \
          --personalized_epochs 0 \
          --local_bs 32 \
          --lora_rank $rank \
          --lora_alpha $((rank * 2)) \
          --fed_algorithm perfedavg \
          --use_lora \
          --device ${DEVICE}
    done
  done
done