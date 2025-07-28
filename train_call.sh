#!/bin/bash

echo "=============================================== Autoformer:milano: call ==============================================="
python main.py \
    --model_type autoformer \
    --file_path milano.h5 \
    --experiment_name autoformer_milano_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== Autoformer:trento: call ==============================================="
python main.py \
    --model_type autoformer \
    --file_path trento.h5 \
    --experiment_name autoformer_trento_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== DLinear:milano: call ==============================================="
python main.py \
    --model_type dLinear \
    --file_path milano.h5 \
    --experiment_name dLinear_milano_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== DLinear:trento: call ==============================================="
python main.py \
    --model_type dLinear \
    --file_path trento.h5 \
    --experiment_name dLinear_trento_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== Informer:milano: call ==============================================="
python main.py \
    --model_type informer \
    --file_path milano.h5 \
    --experiment_name informer_milano_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== Informer:trento: call ==============================================="
python main.py \
    --model_type informer \
    --file_path trento.h5 \
    --experiment_name informer_trento_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== TimeMixer:milano: call ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path milano.h5 \
    --experiment_name timeMixer_milano_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== TimeMixer:trento: call ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path trento.h5 \
    --experiment_name timeMixer_trento_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== TimesNet:milano: call ==============================================="
python main.py \
    --model_type timesNet \
    --file_path milano.h5 \
    --experiment_name timesNet_milano_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2

echo "=============================================== TimesNet:trento: call ==============================================="
python main.py \
    --model_type timesNet \
    --file_path trento.h5 \
    --experiment_name timesNet_trento_call \
    --data_type call \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:2
echo "=== 训练完成 ==="