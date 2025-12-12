#!/bin/bash

# ========== 配置参数 ==========
DATASETS=("milano" "trento") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("net" "call" "sms") #"net" "call" "sms" "traffic"

device="cuda:3"               # 指定使用的 GPU
llm_dim=768
local_ep=1
epoch=10
personalized_epochs=0

# 公共训练参数（所有实验共享）
NUM_CLIENTS=50
seq_len=96
pred_len=24
local_bs=16
fed_algorithm="fedavg"
model_type="timellm"    # 显式定义模型类型，避免未定义变量

# ========== 启动训练循环 ==========
for llm in BERT GPT2; do
  for dataset in "${DATASETS[@]}"; do
      for data_type in "${DATA_TYPES[@]}"; do
          # 构建实验名称
          exp_name="${model_type}_${dataset}_${data_type}_fedavg_${llm}"

          # 执行训练命令
          python main.py \
              --model_type "${model_type}" \
              --file_path "${dataset}.h5" \
              --experiment_name "${exp_name}" \
              --data_type "${data_type}" \
              --llm_model "${llm}" \
              --llm_dim "${llm_dim}" \
              --seq_len "${seq_len}" \
              --pred_len "${pred_len}" \
              --local_ep "${local_ep}" \
              --epoch "${epoch}" \
              --personalized_epochs "${personalized_epochs}" \
              --local_bs "${local_bs}" \
              --fed_algorithm "${fed_algorithm}" \
              --num_clients "${NUM_CLIENTS}" \
              --calculate_communication \
              --device "${device}"
      done
  done
done