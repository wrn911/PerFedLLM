python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_call_perfedavg \
    --data_type call \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_net_perfedavg \
    --data_type net \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_sms_perfedavg \
    --data_type sms \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg \
    --data_type call \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg \
    --data_type net \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg \
    --data_type sms \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3


# deepseek-qwen-1.5B
python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_call_perfedavg_ds \
    --data_type call \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_net_perfedavg_ds \
    --data_type net \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_sms_perfedavg_ds \
    --data_type sms \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_ds \
    --data_type call \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_ds \
    --data_type net \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_ds \
    --data_type sms \
    --llm_model DeepSeek\
    --llm_layers 12\
    --llm_dim 1536\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

#GPT-2 768
python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_call_perfedavg_GPT2 \
    --data_type call \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_net_perfedavg_GPT2 \
    --data_type net \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_sms_perfedavg_GPT2 \
    --data_type sms \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_GPT2 \
    --data_type call \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_GPT2 \
    --data_type net \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_GPT2 \
    --data_type sms \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

#BERT 768
python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_call_perfedavg_BERT \
    --data_type call \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_net_perfedavg_BERT \
    --data_type net \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_sms_perfedavg_BERT \
    --data_type sms \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 5 \
    --epoch 10 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_BERT \
    --data_type call \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_BERT \
    --data_type net \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_BERT \
    --data_type sms \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1