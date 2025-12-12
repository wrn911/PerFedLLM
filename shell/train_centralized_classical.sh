#!/bin/bash

# ================== 集中式训练脚本 (Classical系列) ==================
# 此脚本用于在整个数据集上集中训练Classical模型

# 切换到项目根目录（脚本所在目录的上一级）
cd "$(dirname "$0")"/..

# ================== 配置参数 ==================
# 模型列表
MODELS=("arima" "lasso" "svr" "lstm" "tft" "autoformer" "dLinear" "informer")

# 数据集和数据类型
DATASETS=( "milano" "trento") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("net" "call" "sms") #"net" "call" "sms" "traffic"

# 基础训练参数
SEQ_LEN=96
PRED_LEN=24
EPOCHS=20
BATCH_SIZE=128
GPU_DEVICE="cuda:3"

# ================== 主执行逻辑 ==================
echo "========================================"
echo "    集中式训练脚本 - Classical系列"
echo "========================================"

start_time=$(date +%s)
total_tasks=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#DATA_TYPES[@]}))
current_task=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for data_type in "${DATA_TYPES[@]}"; do
            ((current_task++))
            exp_name="${model}_${dataset}_${data_type}_centralized"

            echo "========== [任务 $current_task/$total_tasks] 正在训练: ${exp_name} =========="

            python main.py \
                --model_type $model \
                --file_path "${dataset}.h5" \
                --experiment_name $exp_name \
                --data_type $data_type \
                --seq_len $SEQ_LEN \
                --pred_len $PRED_LEN \
                --local_bs $BATCH_SIZE \
                --epochs $EPOCHS \
                --training_mode centralized \
                --device $GPU_DEVICE

            if [ $? -eq 0 ]; then
                echo "✓ ${exp_name} 训练成功"
            else
                echo "✗ ${exp_name} 训练失败"
            fi
            echo ""
        done
    done
done

DATASETS=("zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub") #"zte4gup_sub" "zte4gdown_sub" "zte5gup_sub" "zte5gdown_sub"  "milano" "trento"
DATA_TYPES=("traffic") #"net" "call" "sms" "traffic"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for data_type in "${DATA_TYPES[@]}"; do
            ((current_task++))
            exp_name="${model}_${dataset}_${data_type}_centralized"

            echo "========== [任务 $current_task/$total_tasks] 正在训练: ${exp_name} =========="

            python main.py \
                --model_type $model \
                --file_path "${dataset}.h5" \
                --experiment_name $exp_name \
                --data_type $data_type \
                --seq_len $SEQ_LEN \
                --pred_len $PRED_LEN \
                --local_bs $BATCH_SIZE \
                --epochs $EPOCHS \
                --training_mode centralized \
                --device $GPU_DEVICE

            if [ $? -eq 0 ]; then
                echo "✓ ${exp_name} 训练成功"
            else
                echo "✗ ${exp_name} 训练失败"
            fi
            echo ""
        done
    done
done

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "========================================"
echo "      Classical系列 训练已完成"
echo "========================================"
echo "总用时: ${minutes}分${seconds}秒"
echo "========================================"
