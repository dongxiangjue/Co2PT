#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


for seed in "${seeds[@]}"
do
python run_prompt_cl_bios.py \
    --pre_seq_len ${prompt} \
    --hidden_dropout_prob 0.1 \
    --model_name_or_path bert-base-uncased \
    --max_seq_len $max_length \
    --batch_size $batch \
    --align_temp ${align} \
    --temp ${temp} \
    --lr $learning \
    --num_epochs $epoch \
    --cl_loss \
    --ckpt_dir $dir} \
    --seed $seed \
done