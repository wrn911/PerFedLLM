#!/bin/bash

# ================== 通用集中式训练脚本 ==================
# 此脚本用于在整个数据集上集中训练模型，支持TimeLLM和Classical模型

# ================== 配置参数 ==================
# 模型列表 - 您可以根据需要添加或删除模型
# TimeLLM系列: "TimeLLM", "SimpleTimeLLM"
# Classical系列: "DLinear", "Autoformer", "Informer", "TimesNet", "LSTM", "TFT"
MODELS=("DLinear" "Autoformer" "SimpleTimeLLM")

# 数据集和数据类型 - 您可以根据需要进行修改
DATASETS=("milano")
DATA_TYPES=("net" "call")

# 基础训练参数
SEQ_LEN=96
PRED_LEN=24
EPOCHS=20
BATCH_SIZE=128 # 集中式训练可以使用更大的批次大小
GPU_DEVICE="cuda:2"  # 根据您的设置修改GPU

# TimeLLM系列专属参数
LLM_MODEL="GPT2"    # 可选: "BERT", "GPT2"
LLM_DIM=768         # BERT: 768, GPT2: 768
LORA_RANK=16
LORA_ALPHA=32

# ================== 主执行逻辑 ==================
echo "========================================"
echo "         集中式模型训练脚本"
echo "========================================"
echo "GPU设备: $GPU_DEVICE"
echo "序列长度: $SEQ_LEN, 预测长度: $PRED_LEN"
echo "========================================"
echo ""

# 记录开始时间
start_time=$(date +%s)
total_tasks=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#DATA_TYPES[@]}))
current_task=0

# 循环训练
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for data_type in "${DATA_TYPES[@]}"; do
            ((current_task++))
            exp_name="${model}_${dataset}_${data_type}_centralized"

            echo "========== [任务 $current_task/$total_tasks] 正在训练: ${exp_name} =========="

            # 基础命令
            CMD="python main.py \
                --model_type $model \
                --file_path "${dataset}.h5" \
                --experiment_name $exp_name \
                --data_type $data_type \
                --seq_len $SEQ_LEN \
                --pred_len $PRED_LEN \
                --local_bs $BATCH_SIZE \
                --epochs $EPOCHS \
                --training_mode centralized \
                --device $GPU_DEVICE"

            # 根据模型类型添加特定参数
            if [[ "$model" == "TimeLLM" || "$model" == "SimpleTimeLLM" ]]; then
                # 这是TimeLLM系列模型，添加LLM和LoRA相关参数
                CMD+=" --llm_model $LLM_MODEL \
                        --llm_dim $LLM_DIM \
                        --use_lora \
                        --lora_rank $LORA_RANK \
                        --lora_alpha $LORA_ALPHA"
                echo "模型类型: TimeLLM (已添加LLM和LoRA参数)"
            else
                # 这是Classical模型，无需额外参数
                echo "模型类型: Classical"
            fi

            # 执行训练命令
            eval $CMD

            if [ $? -eq 0 ]; then
                echo "✓ ${exp_name} 训练成功"
            else
                echo "✗ ${exp_name} 训练失败"
            fi
            echo ""
        done
    done
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "========================================"
echo "           所有训练已完成"
echo "========================================"
echo "总用时: ${minutes}分${seconds}秒"
echo "完成任务数: $total_tasks"
echo "========================================"