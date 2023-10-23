#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

for seed in "${seeds[@]}"
do
python run_base.py \
      --checkpointing_steps epoch \
      --model_name_or_path ${model} \
      --task_name ${task} \
      --max_length 128 \
      --per_device_train_batch_size ${batch} \
      --learning_rate ${learning} \
      --num_train_epochs ${epoch} \
      --seed ${seed} \
      --early_stopping ${early} \
      --output_dir ./${date}_base/${task}/${tag}/${seed}/${learning}_${batch} \
done
