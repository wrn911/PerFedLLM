#!/bin/bash

# ================== 简化的经典模型训练脚本 ==================
# 训练ARIMA, Lasso, SVR, Prophet, LSTM, TFT等经典模型

# ================== 配置参数 ==================
# 模型列表 - 可以根据需要修改
MODELS=("arima" "lasso" "svr" "lstm" "tft")

# 数据集和数据类型 - 可以根据需要修改
DATASETS=("milano" "trento")
DATA_TYPES=("net" "call" "sms")

# 训练参数
SEQ_LEN=96
PRED_LEN=24
EPOCHS=20
GPU_DEVICE="cuda:3"  # 可以修改为 cuda:1, cuda:2 等

# ================== 主要函数 ==================

# 训练单个模型
train_model() {
    local model=$1
    local dataset=$2
    local data_type=$3

    local exp_name="${model}_${dataset}_${data_type}"

    echo "========== 训练: ${exp_name} =========="

    # 根据模型类型设置特定参数
    local extra_params=""
    case $model in
        "arima"|"lasso"|"svr"|"prophet")
            extra_params="--epochs 1"  # 传统模型不需要多轮训练
            ;;
        "lstm"|"tft")
            extra_params="--epochs $EPOCHS"  # 神经网络模型需要多轮训练
            ;;
    esac

    # 执行训练命令
    python main.py \
        --model_type $model \
        --file_path ${dataset}.h5 \
        --experiment_name $exp_name \
        --data_type $data_type \
        --seq_len $SEQ_LEN \
        --pred_len $PRED_LEN \
        --training_mode distributed \
        --device $GPU_DEVICE \
        $extra_params

    if [ $? -eq 0 ]; then
        echo "✓ ${exp_name} 训练成功"
    else
        echo "✗ ${exp_name} 训练失败"
    fi
    echo ""
}

# 显示帮助信息
show_help() {
    echo "简化的经典模型训练脚本"
    echo ""
    echo "用法:"
    echo "  $0                    # 训练所有模型"
    echo "  $0 lstm              # 只训练LSTM模型"
    echo "  $0 arima lasso       # 训练ARIMA和Lasso模型"
    echo ""
    echo "支持的模型: ${MODELS[*]}"
    echo "数据集: ${DATASETS[*]}"
    echo "数据类型: ${DATA_TYPES[*]}"
    echo ""
    echo "修改配置:"
    echo "  编辑脚本顶部的配置参数部分"
    echo ""
}

# ================== 主执行逻辑 ==================

main() {
    echo "========================================"
    echo "       经典模型训练脚本"
    echo "========================================"
    echo "GPU设备: $GPU_DEVICE"
    echo "序列长度: $SEQ_LEN, 预测长度: $PRED_LEN"
    echo "========================================"
    echo ""

    # 确定要训练的模型
    local models_to_train=()

    if [ $# -eq 0 ]; then
        # 没有参数，训练所有模型
        models_to_train=("${MODELS[@]}")
        echo "训练所有模型: ${models_to_train[*]}"
    else
        # 有参数，只训练指定的模型
        for arg in "$@"; do
            if [[ " ${MODELS[*]} " =~ " $arg " ]]; then
                models_to_train+=("$arg")
            else
                echo "警告: 不支持的模型 '$arg'，跳过"
            fi
        done

        if [ ${#models_to_train[@]} -eq 0 ]; then
            echo "错误: 没有有效的模型被指定"
            echo "支持的模型: ${MODELS[*]}"
            return 1
        fi

        echo "训练指定模型: ${models_to_train[*]}"
    fi

    echo ""

    # 记录开始时间
    local start_time=$(date +%s)
    local total_tasks=$((${#models_to_train[@]} * ${#DATASETS[@]} * ${#DATA_TYPES[@]}))
    local current_task=0

    echo "开始训练，总共 $total_tasks 个任务"
    echo ""

    # 循环训练
    for model in "${models_to_train[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for data_type in "${DATA_TYPES[@]}"; do
                ((current_task++))
                echo "进度: $current_task/$total_tasks"
                train_model $model $dataset $data_type
            done
        done
    done

    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo "========================================"
    echo "           训练完成"
    echo "========================================"
    echo "总用时: ${minutes}分${seconds}秒"
    echo "完成任务: $total_tasks"
    echo "========================================"
}

# ================== 入口点 ==================

# 检查帮助参数
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# 简单检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "错误: 未找到main.py文件"
    exit 1
fi

# 运行主程序
main "$@"