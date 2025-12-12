#!/bin/bash

FILE_PATH="trento.h5"
DEVICE="cuda:1"
TRAIN_RATIO=0.1

# ========== simpletimellm 系列 ==========
for LLM in BERT GPT2 Qwen3; do
  for DATA in call net sms; do
    EXP_NAME="simpletimellm_trento_${DATA}_perfedavg_${LLM}_10fewshot"

    CMD="python main.py \
      --model_type simpletimellm \
      --file_path ${FILE_PATH} \
      --experiment_name ${EXP_NAME} \
      --data_type ${DATA} \
      --seq_len 96 \
      --pred_len 24 \
      --local_ep 10 \
      --epoch 20 \
      --train_ratio ${TRAIN_RATIO} \
      --personalized_epochs 0 \
      --local_bs 32 \
      --lora_rank 16 \
      --lora_alpha 32 \
      --fed_algorithm perfedavg \
      --use_lora \
      --device ${DEVICE}"

    # BERT/GPT2 需要额外指定 LLM 参数
    if [ "$LLM" != "Qwen3" ]; then
      CMD+=" --llm_model ${LLM} --llm_dim 768"
    fi

    echo "Running ${EXP_NAME}..."
    eval $CMD
  done
done

# ========== autoformer ==========
for DATA in call net sms; do
  python main.py \
    --model_type autoformer \
    --file_path ${FILE_PATH} \
    --experiment_name autoformer_trento_${DATA}_10fewshot \
    --data_type ${DATA} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --epochs 0 \
    --train_ratio ${TRAIN_RATIO} \
    --personalized_epochs 20 \
    --local_bs 64 \
    --device ${DEVICE}
done

# ========== tft ==========
for DATA in call net sms; do
  python main.py \
    --model_type tft \
    --file_path ${FILE_PATH} \
    --experiment_name tft_trento_${DATA}_10fewshot \
    --data_type ${DATA} \
    --seq_len 96 \
    --pred_len 24 \
    --training_mode distributed \
    --device ${DEVICE} \
    --train_ratio ${TRAIN_RATIO} \
    --epochs 20
done

# ========== dLinear ==========
for DATA in call net sms; do
  python main.py \
    --model_type dLinear \
    --file_path ${FILE_PATH} \
    --experiment_name dLinear_trento_${DATA}_10fewshot \
    --data_type ${DATA} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --epochs 0 \
    --train_ratio ${TRAIN_RATIO} \
    --personalized_epochs 20 \
    --local_bs 64 \
    --device ${DEVICE}
done
