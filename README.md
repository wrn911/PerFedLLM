# 总体实验

## 经典方法

## 时序预测方法

## our

# 消融实验

```
======BERT=========
python main.py \
    --model_type simpletimellm \
    --file_path milano.h5 \
    --experiment_name simpletimellm_milano_call_perfedavg_BERT_NoPrompt \
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
    --experiment_name simpletimellm_milano_net_perfedavg_BERT_NoPrompt \
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
    --experiment_name simpletimellm_milano_sms_perfedavg_BERT_NoPrompt \
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

```

# 少样本实验

## 5%样本

```
======BERT=========
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_BERT_5fewshot \
    --data_type call \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_BERT_5fewshot \
    --data_type net \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_BERT_5fewshot \
    --data_type sms \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

```

```
======GPT2=============
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_GPT2_5fewshot \
    --data_type call \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_GPT2_5fewshot \
    --data_type net \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_GPT2_5fewshot \
    --data_type sms \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1
```

```
=========Qwen=============
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_qwen_5fewshot \
    --data_type call \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_qwen_5fewshot \
    --data_type net \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_qwen_5fewshot \
    --data_type sms \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.05 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

```

```
=========autoformer=============
python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_call_5fewshot \
            --data_type call \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:0

python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_net_5fewshot \
            --data_type net \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:1

python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_sms_5fewshot \
            --data_type sms \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:2

```

```
=========tft=============
    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_call_5fewshot \
        --data_type call \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:0 \
    	--train_ratio 0.05 \
        --epochs 20 

    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_net_5fewshot \
        --data_type net \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:1\
    	--train_ratio 0.05 \
        --epochs 20 

    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_sms_5fewshot \
        --data_type sms \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:2 \
    	--train_ratio 0.05 \
        --epochs 20 

```

```
=========dLinear=============
python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_call_5fewshot \
            --data_type call \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_net_5fewshot \
            --data_type net \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_sms_5fewshot \
            --data_type sms \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.05 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

```

## 10%样本

```
======BERT=========
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_BERT_10fewshot \
    --data_type call \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_BERT_10fewshot \
    --data_type net \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_BERT_10fewshot \
    --data_type sms \
    --llm_model BERT\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

```

```
======GPT2=============
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_GPT2_10fewshot \
    --data_type call \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_GPT2_10fewshot \
    --data_type net \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_GPT2_10fewshot \
    --data_type sms \
    --llm_model GPT2\
    --llm_dim 768\
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:1
```

```
=========Qwen=============
python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_call_perfedavg_qwen_10fewshot \
    --data_type call \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:2

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_net_perfedavg_qwen_10fewshot \
    --data_type net \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:3

python main.py \
    --model_type simpletimellm \
    --file_path trento.h5 \
    --experiment_name simpletimellm_trento_sms_perfedavg_qwen_10fewshot \
    --data_type sms \
    --seq_len 96 \
    --pred_len 24 \
    --local_ep 10 \
    --epoch 20 \
    --train_ratio 0.1 \
    --personalized_epochs 0 \
    --local_bs 32 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --fed_algorithm perfedavg \
    --device cuda:0

```

```
=========autoformer=============
python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_call_10fewshot \
            --data_type call \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:0

python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_net_10fewshot \
            --data_type net \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:1

python main.py \
            --model_type autoformer \
            --file_path trento.h5 \
            --experiment_name autoformer_trento_sms_10fewshot \
            --data_type sms \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:2

```

```
=========tft=============
    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_call_10fewshot \
        --data_type call \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:0 \
    	--train_ratio 0.1 \
        --epochs 20 

    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_net_10fewshot \
        --data_type net \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:1\
    	--train_ratio 0.1 \
        --epochs 20 

    python main.py \
        --model_type tft \
        --file_path trento.h5 \
        --experiment_name tft_trento_sms_10fewshot \
        --data_type sms \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:2 \
    	--train_ratio 0.1 \
        --epochs 20 

```
```
=========dLinear=============
python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_call_10fewshot \
            --data_type call \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_net_10fewshot \
            --data_type net \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

python main.py \
            --model_type dLinear \
            --file_path trento.h5 \
            --experiment_name dLinear_trento_sms_10fewshot \
            --data_type sms \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 24 \
            --local_ep 0 \
            --epochs 0 \
    	    --train_ratio 0.1 \
            --personalized_epochs 20 \
            --local_bs 64 \
            --device cuda:3

```


# 零样本实验

```
./compare_models.sh
```

```
=========autoformer=============
    python main.py \
        --model_type autoformer \
        --file_path trento.h5 \
        --experiment_name autoformer_trento_call_zeroshot \
        --data_type call \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:0 \
        --epochs 20 

    python main.py \
        --model_type autoformer \
        --file_path trento.h5 \
        --experiment_name autoformer_trento_net_zeroshot \
        --data_type net \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:1\
        --epochs 20 

    python main.py \
        --model_type autoformer \
        --file_path trento.h5 \
        --experiment_name autoformer_trento_sms_zeroshot \
        --data_type sms \
        --seq_len 96 \
        --pred_len 24 \
        --training_mode distributed \
        --device cuda:2 \
        --epochs 20 
```

```
=========dLinear=============
        python main.py \
            --model_type dLinear \
            --file_path "milano.h5" \
            --experiment_name dLinear_milano_call \
            --data_type call \
        --seq_len 96 \
        --pred_len 24 \
            --local_ep 0 \
            --epoch 0 \
            --personalized_epochs 20 \
            --local_bs 64 \
        --device cuda:0 
        
        python main.py \
            --model_type dLinear \
            --file_path "milano.h5" \
            --experiment_name dLinear_milano_net \
            --data_type net \
        --seq_len 96 \
        --pred_len 24 \
            --local_ep 0 \
            --epoch 0 \
            --personalized_epochs 20 \
            --local_bs 64 \
        --device cuda:1
        
                python main.py \
            --model_type dLinear \
            --file_path "milano.h5" \
            --experiment_name dLinear_milano_sms \
            --data_type sms \
        --seq_len 96 \
        --pred_len 24 \
            --local_ep 0 \
            --epoch 0 \
            --personalized_epochs 20 \
            --local_bs 64 \
        --device cuda:2
```

```

```





# 绘制样本

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/milano_call_predictions/client_1539.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_milano_call_predictions/client_1539.json \
--name2 DLinear \
--sample_id 25 \
--data_type denormalized \
--output ./results/milano_call.png
```

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/milano_net_predictions/client_6960.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_milano_net_predictions/client_6960.json \
--name2 DLinear \
--sample_id 5 \
--data_type denormalized \
--output ./results/milano_net.png
```

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/milano_sms_predictions/client_3277.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_milano_sms_predictions/client_3277.json \
--name2 DLinear \
--sample_id 10 \
--data_type denormalized \
--output ./results/milano_sms.png
```

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/trento_call_predictions/client_4744.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_trento_call_predictions/client_4744.json \
--name2 DLinear \
--sample_id 10 \
--data_type denormalized \
--output ./results/trento_call.png
```

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/trento_net_predictions/client_5073.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_trento_net_predictions/client_5073.json \
--name2 DLinear \
--sample_id 15 \
--data_type denormalized \
--output ./results/trento_net.png
```

```
python ./draw/visualize_comparison.py \
--file1 ./detailed_data/trento_sms_predictions/client_2738.json \
--name1 FedLLM-WTP \
--file2 ./detailed_data/dLinear_trento_sms_predictions/client_2738.json \
--name2 DLinear \
--sample_id 5 \
--data_type denormalized \
--output ./results/trento_sms.png
```

