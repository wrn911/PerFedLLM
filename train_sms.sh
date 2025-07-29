#!/bin/bash

# ================== 可配置参数 ==================
# 定义模型列表、数据集列表和联邦学习算法列表
MODELS=("autoformer" "dLinear" "informer" "timeMixer" "timesNet")
DATASETS=("milano" "trento")
FED_ALGORITHMS=("fedavg" "fedprox" "perfedavg")
GPU_DEVICE="cuda:3" # 指定使用的GPU

# 定义公共参数
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=24
LOCAL_BS=64

# ================== 实验执行 ==================

echo "----------------------- 1. 本地独立训练 (基线) ------------------------------"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        exp_name="${model}_${dataset}_sms"
        echo "================ Running: ${exp_name} ================"
        python main.py \
            --model_type "$model" \
            --file_path "${dataset}.h5" \
            --experiment_name "$exp_name" \
            --data_type sms \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --local_ep 0 \
            --epoch 0 \
            --personalized_epochs 20 \
            --local_bs $LOCAL_BS \
            --device $GPU_DEVICE
    done
done

echo ""
echo "----------------------- 2. 联邦学习训练 ------------------------------"
for algo in "${FED_ALGORITHMS[@]}"; do
    echo "----------------------- Running Federated Algorithm: $algo ------------------------------"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            exp_name="${model}_${dataset}_sms_${algo}"
            echo "================ Running: ${exp_name} ================"

            # 构造参数
            params=(
                --model_type "$model"
                --file_path "${dataset}.h5"
                --experiment_name "$exp_name"
                --data_type sms
                --seq_len $SEQ_LEN
                --label_len $LABEL_LEN
                --pred_len $PRED_LEN
                --fed_algorithm "$algo"
                --local_bs $LOCAL_BS
                --device $GPU_DEVICE
            )

            # Per-FedAvg不需要personalized_epochs参数，其他需要设置为0
            if [ "$algo" != "perfedavg" ]; then
                params+=(--personalized_epochs 0)
            fi

            python main.py "${params[@]}"
        done
    done
done


echo "=== 所有训练完成 ==="