#!/bin/bash

echo "=============================================== Autoformer:milano: sms ==============================================="
python main.py \
    --model_type autoformer \
    --file_path milano.h5 \
    --experiment_name autoformer_milano_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== Autoformer:trento: sms ==============================================="
python main.py \
    --model_type autoformer \
    --file_path trento.h5 \
    --experiment_name autoformer_trento_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== DLinear:milano: sms ==============================================="
python main.py \
    --model_type dLinear \
    --file_path milano.h5 \
    --experiment_name dLinear_milano_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== DLinear:trento: sms ==============================================="
python main.py \
    --model_type dLinear \
    --file_path trento.h5 \
    --experiment_name dLinear_trento_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== Informer:milano: sms ==============================================="
python main.py \
    --model_type informer \
    --file_path milano.h5 \
    --experiment_name informer_milano_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== Informer:trento: sms ==============================================="
python main.py \
    --model_type informer \
    --file_path trento.h5 \
    --experiment_name informer_trento_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== TimeMixer:milano: sms ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path milano.h5 \
    --experiment_name timeMixer_milano_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== TimeMixer:trento: sms ==============================================="
python main.py \
    --model_type timeMixer \
    --file_path trento.h5 \
    --experiment_name timeMixer_trento_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== TimesNet:milano: sms ==============================================="
python main.py \
    --model_type timesNet \
    --file_path milano.h5 \
    --experiment_name timesNet_milano_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3

echo "=============================================== TimesNet:trento: sms ==============================================="
python main.py \
    --model_type timesNet \
    --file_path trento.h5 \
    --experiment_name timesNet_trento_sms \
    --data_type sms \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --local_ep 0 \
    --personalized_epochs 20 \
    --epoch 0 \
    --local_bs 64 \
    --device cuda:3
echo "=== 训练完成 ==="