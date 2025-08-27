#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed  train_sft2.py \
    --model_name_or_path model_checkpoint/cloud_sft1_0825 \
    --dataset_name data/sky_selfgen_critique.json \
    --cutoff_len 2048 \
    --output_dir save/cloud_sft2_0825 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --deepspeed deepspeed/ds_z3_config.json \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --report_to wandb \
    --run_name cloud_sft2_0825