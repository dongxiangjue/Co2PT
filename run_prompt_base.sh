#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python run_prompt_base.py \
      --pre_seq_len ${prompt} \
      --checkpointing_steps epoch \
      --hidden_dropout_prob 0.1 \
      --model_name_or_path bert-base-uncased \
      --task_name ${task} \
      --max_length 128 \
      --per_device_train_batch_size ${batch} \
      --learning_rate ${learning} \
      --num_train_epochs ${epoch} \
      --seed ${seed} \
      --early_stopping ${early} \
      --output_dir ./${date}_promptbase/${task}/${seed}/${learning}_${batch}_${prompt}_${epoch} \
