#!/bin/bash

export ROOT_PATH=/Users/zejun7/python/llm-training

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name ${ROOT_PATH}/reward_model/original_data/Skywork-Reward-Preference-80K-v0.1/data/train-00000-of-00001.parquet \
    --output_dir ${ROOT_PATH}/reward_model/model/base_reward/save/base_reward_0926 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_only_model \
    --max_length 2048 \
    --deepspeed ${ROOT_PATH}/deepspeed/ds_z3_config.json \
    --report_to wandb \
    --run_name base_reward_0926