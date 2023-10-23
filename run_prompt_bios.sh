#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

for seed in "${seeds[@]}"
do
python run_prompt_bios.py \
    --pre_seq_len ${prompt} \
    --hidden_dropout_prob 0.1 \
    --model_name_or_path bert-base-uncased \
    --max_seq_len $max_length \
    --batch_size $batch \
    --lr $learning \
    --num_epochs $epoch \
    --ckpt_dir $dir \
    --seed $seed \
done