# Co$^2$PT: Mitigating Bias in Pre-trained Language Models through Counterfactual Contrastive Prompt Tuning
Anonymous code repository for paper submission.

## Dataset and Extrinsic Benchmarks:

**STS-B** and **SNLI** datasets are from the Hugging Face Datasets library: https://github.com/huggingface/datasets. Apache License 2.0.

**Bias-STS-B** is from Lauscher et al. (2021) and **Bias-NLI** is from He et al. (2022).

**Bias-in-Bios** is downloaded through https://github.com/shauli-ravfogel/nullspace_projection/blob/master/download_data.sh. MIT License.

## Debiased Baseline Models

**ZariCDA, ZariDO**: The checkpoints of these two models are from https://github.com/google-research-datasets/Zari. Apache-2.0 license.

**Context-Debias**: The checkpoints are from https://github.com/kanekomasahiro/context-debias. MIT license.

**Auto-Debias**: The checkpoints are from https://github.com/Irenehere/Auto-Debias.

**MABEL**: The checkpoints are from https://huggingface.co/princeton-nlp/mabel-bert-base-uncased. MIT License.

## Fine-tuning Baselines
For STS-B and SNLI datasets: 
```
chmod +x run_base.sh
./run_base.sh
```
For Bias-in-Bios dataset:
```
chmod +x run_base_bios.sh
./run_base_bios.sh
```
## Prompt Tuning
For STS-B and SNLI datasets: 
```
chmod +x run_prompt_base.sh
./run_prompt_base.sh
```
For Bias-in-Bios dataset:
```
chmod +x run_prompt_bios.sh
./run_prompt_bios.sh
```

## Co$^2$PT
For STS-B and SNLI datasets: 
```
chmod +x run_prompt_cl.sh
./run_prompt_cl.sh
```
For Bias-in-Bios dataset:
```
chmod +x run_prompt_cl_bios.sh
./run_prompt_cl_bios.sh
```

## Evaluation
```
python eval_stsb_bias.py --model_name_or_path ${model}
python eval_stsb_bias_prompt.py --model_name_or_path ${model}

python eval_nli_bias.py --model_name_or_path ${model} --batch_size 32
python eval_nli_bias_prompt.py --model_name_or_path ${model} --batch_size 64

python eval_bios.py --model_name_or_path "bert-base-uncased" --load_from_file ${model}
python eval_bios_prompt.py --pre_seq_len ${prompt} --hidden_dropout_prob 0.1 --model_name_or_path "bert-base-uncased" --load_from_file ${model}
```

## Code Acknowledgements
- Evaluation code for Bias-NLI and Bias-in-Bios is adapted from He et al. (2022).
- Evaluation code for Bias-STS-B is adapted from Lauscher et al. (2021).
- Fine-tuning and prompt tuning code rely on the Huggingface implementation.

