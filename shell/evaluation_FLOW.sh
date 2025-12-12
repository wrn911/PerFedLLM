#!/bin/bash

# 切换到项目根目录（脚本所在目录的上一级）
cd "$(dirname "$0")"/..

# ========== 配置参数 ==========
DATASETS=( "milano" "trento") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("net" "call" "sms") #"net" "call" "sms" "traffic"
device="cuda:3"               # 指定使用的 GPU


# ========== 启动循环 ==========
for dataset in "${DATASETS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
         python test_evaluation.py \
           --model_path "experiments/simpletimellm_${dataset}_${data_type}_perfedavg_BERT/global_model.pth" \
           --config_path "experiments/simpletimellm_${dataset}_${data_type}_perfedavg_BERT/config.json" \
           --data_file "${dataset}.h5" \
           --data_type "${data_type}" \
           --save_predictions_csv "results/predictions/${dataset}_${data_type}_predictions.csv" \
           --device "${device}"
    done
done

DATASETS=("zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("traffic") #"net" "call" "sms" "traffic"

# ========== 启动循环 ==========
for dataset in "${DATASETS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
         python test_evaluation.py \
           --model_path "experiments/simpletimellm_${dataset}_${data_type}_perfedavg_BERT/global_model.pth" \
           --config_path "experiments/simpletimellm_${dataset}_${data_type}_perfedavg_BERT/config.json" \
           --data_file "${dataset}.h5" \
           --data_type "${data_type}" \
           --save_predictions_csv "results/predictions/${dataset}_${data_type}_predictions.csv" \
           --device "${device}"
    done
done