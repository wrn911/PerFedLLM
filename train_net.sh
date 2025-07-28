#!/bin/bash

echo "=============================================== Autoformer:milano: net ==============================================="
python main.py \
    --model_type autoformer \
    --file_path milano.h5 \
    --experiment_name autoformer_milano_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== Autoformer:trento: net ==============================================="
python main.py \
    --model_type autoformer \
    --file_path trento.h5 \
    --experiment_name autoformer_trento_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== DLinear:milano: net ==============================================="
python main.py \
    --model_type dLinear \
    --file_path milano.h5 \
    --experiment_name dLinear_milano_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== DLinear:trento: net ==============================================="
python main.py \
    --model_type dLinear \
    --file_path trento.h5 \
    --experiment_name dLinear_trento_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== Informer:milano: net ==============================================="
python main.py \
    --model_type informer \
    --file_path milano.h5 \
    --experiment_name informer_milano_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== Informer:trento: net ==============================================="
python main.py \
    --model_type informer \
    --file_path trento.h5 \
    --experiment_name informer_trento_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== TimeMixer:milano: net ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path milano.h5 \
    --experiment_name timeMixer_milano_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== TimeMixer:trento: net ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path trento.h5 \
    --experiment_name timeMixer_trento_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== TimesNet:milano: net ==============================================="
python main.py \
    --model_type timesNet \
    --file_path milano.h5 \
    --experiment_name timesNet_milano_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1

echo "=============================================== TimesNet:trento: net ==============================================="
python main.py \
    --model_type timesNet \
    --file_path trento.h5 \
    --experiment_name timesNet_trento_net \
    --data_type net \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:1
echo "=== 训练完成 ==="