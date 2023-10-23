#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

option=2 # 1-align_loss; 2-cl_loss; 3-align_loss+avg; 4-cl_loss+avg; 5-cl_loss for nli; 6-cl_loss plus self_cl_loss 
model=bert-base-uncased   # bert-base-uncased


if [ ${option} = 1 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --output_dir ./${date}_cl/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_cl_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &

elif [ ${option} = 2 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --cl_loss \
        --output_dir ./${date}_clcl/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_clcl_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &

elif [ ${option} = 3 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --use_average \
        --output_dir ./${date}_clavg/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_clavg_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &

elif [ ${option} = 4 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --cl_loss \
        --use_average \
        --output_dir ./${date}_clclavg/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_clclavg_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &

elif [ ${option} = 5 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --cl_loss \
        --train_nli \
        --output_dir ./${date}_clclnli/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_clclnli_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &

elif [ ${option} = 6 ]; then
  nohup python run_prompt_cl.py \
        --pre_seq_len ${prompt} \
        --checkpointing_steps epoch \
        --hidden_dropout_prob 0.1 \
        --model_name_or_path ${model} \
        --task_name ${task} \
        --max_length 128 \
        --per_device_train_batch_size ${batch} \
        --learning_rate ${learning} \
        --num_train_epochs ${epoch} \
        --align_temp ${align} \
        --temp ${temp} \
        --seed ${seed} \
        --bias_type ${type} \
        --early_stopping ${early} \
        --cl_loss \
        --plus_self \
        --output_dir ./${date}_clclplusself/${task}/${type}/${seed}/${align}_${temp}_${learning}_${prompt}_${batch}_${epoch} \
        >"logs/${date}_clclplusself_${task}_${type}_${seed}_${align}_${temp}_${learning}_${prompt}_${batch}_${epoch}.log" 2>&1 &
fi








