#!/bin/bash

# ========== 配置参数 ==========
DATASETS=("milano" "trento") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("net" "call" "sms") #"net" "call" "sms" "traffic"

device="cuda:0"               # 指定使用的 GPU
llm="BERT"                    # 可选: "BERT", "GPT2", "LLAMA"
llm_dim=768                   # 768 768 4096
local_ep=0
epoch=0
personalized_epochs=20

# 公共训练参数（所有实验共享）
NUM_CLIENTS=50
seq_len=96
pred_len=24
local_bs=16
fed_algorithm="perfedavg"
model_type="timellm"    # 显式定义模型类型，避免未定义变量

# ========== 启动训练循环 ==========
for dataset in "${DATASETS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        # 构建实验名称
        exp_name="${model_type}_${dataset}_${data_type}_${llm}"

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
            --device "${device}"
    done
done