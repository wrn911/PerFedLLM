#!/bin/bash

# ================== 集中式训练脚本 (Classical系列) ==================
# 此脚本用于在整个数据集上集中训练Classical模型。
# 它会自动区分神经网络模型和传统统计模型，并调用相应的训练器。

# ================== 配置参数 ==================
# 神经网络模型 (使用 main.py -> centralized_trainer.py)
CLASSICAL_NN_MODELS=("DLinear" "Autoformer" "Informer" "TimesNet" "LSTM")

# 传统统计模型 (使用 classical_trainer.py)
CLASSICAL_STAT_MODELS=("ARIMA" "SVR" "Lasso")

# 将两个列表合并用于循环
MODELS=("${CLASSICAL_NN_MODELS[@]}" "${CLASSICAL_STAT_MODELS[@]}")

# 数据集和数据类型
DATASETS=("milano")
DATA_TYPES=("net" "call")

# 基础训练参数
SEQ_LEN=96
PRED_LEN=24
EPOCHS=20
BATCH_SIZE=128
GPU_DEVICE="cuda:2"

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

            # 检查是否为传统统计模型
            is_stat_model=0
            for stat_model in "${CLASSICAL_STAT_MODELS[@]}"; do
                if [[ "$model" == "$stat_model" ]]; then
                    is_stat_model=1
                    break
                fi
            done

            if [ $is_stat_model -eq 1 ]; then
                # --- 调用真正的集中式传统统计模型训练器 ---
                echo "模型类型: 传统统计模型. 调用 train_classical_centralized.py"
                python train_classical_centralized.py \
                    --model_type $model \
                    --file_path "${dataset}.h5" \
                    --data_type $data_type \
                    --seq_len $SEQ_LEN \
                    --pred_len $PRED_LEN \
                    --experiment_name $exp_name
            else
                # --- 调用基于神经网络的集中式训练器 ---
                echo "模型类型: 神经网络模型. 调用 main.py (centralized mode)"
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
            fi

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
