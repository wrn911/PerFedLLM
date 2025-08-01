#!/bin/bash
types=("net" "call" "sms")
for type in "${types[@]}"; do
    echo "Evaluating $type..."
    python test_evaluation.py \
        --model_path ./experiments/simpletimellm_milano_${type}_perfedavg/global_model.pth \
        --config_path ./experiments/simpletimellm_milano_${type}_perfedavg/config.json \
        --save_results ./results/Qwen3_milano2trento_${type}_metrics.json \
        --data_file trento.h5 \
        --data_type ${type} \
        --device cuda:3
    python test_evaluation.py \
        --model_path ./experiments/simpletimellm_milano_${type}_perfedavg_GPT2/global_model.pth \
        --config_path ./experiments/simpletimellm_milano_${type}_perfedavg_GPT2/config.json \
        --save_results ./results/GPT_milano2trento_${type}_metrics.json \
        --data_file trento.h5 \
        --data_type ${type} \
        --device cuda:3
    python test_evaluation.py \
        --model_path ./experiments/simpletimellm_milano_${type}_perfedavg_BERT/global_model.pth \
        --config_path ./experiments/simpletimellm_milano_${type}_perfedavg_BERT/config.json \
        --save_results ./results/BERT_milano2trento_${type}_metrics.json \
        --data_file trento.h5 \
        --data_type ${type} \
        --device cuda:3
done