#!/bin/bash

FILE_PATH="milano.h5"
DEVICE="cuda:0"

echo "----------------------- 消融实验：删除联邦学习------------------------------"
for DATA in call net sms; do
  EXP_NAME="simpletimellm_milano_${DATA}           "
  python main.py \
    --model_type simpletimellm \
    --file_path ${FILE_PATH} \
    --experiment_name ${EXP_NAME} \
    --llm_model BERT\
    --llm_dim 768\
    --data_type ${DATA} \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 0 \
    --epoch 0 \
    --personalized_epochs 20 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --use_lora \
    --device ${DEVICE}
done

echo "----------------------- 消融实验：删除LoRA------------------------------"
for DATA in call net sms; do
  EXP_NAME="simpletimellm_milano_${DATA}_perfedavg_BERT_NoLoRA"
  python main.py \
    --model_type simpletimellm \
    --file_path ${FILE_PATH} \
    --experiment_name ${EXP_NAME} \
    --llm_model BERT\
    --llm_dim 768\
    --data_type ${DATA} \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device ${DEVICE}
done

echo "----------------------- 消融实验：删除提示词------------------------------"
for DATA in call net sms; do
  EXP_NAME="simpletimellm_milano_${DATA}_perfedavg_BERT_NoPrompt"
  python main.py \
    --model_type simpletimellm \
    --file_path ${FILE_PATH} \
    --experiment_name ${EXP_NAME} \
    --llm_model BERT\
    --llm_dim 768\
    --data_type ${DATA} \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --use_lora \
    --no_prompt \
    --device ${DEVICE}
done
