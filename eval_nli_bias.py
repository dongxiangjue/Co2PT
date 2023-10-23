import argparse
from argparse import Namespace
import os

import logging
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import json

from datasets import load_dataset

from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def evaluate_bias_nli(model, args, eval_loader, output_dir, model_path):

    nn_count, fn_count, tn_count, tn2_count, denom = 0, 0, 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(eval_loader)):

        input_ids = batch["input_ids"].to(args.device)

        attention_mask = batch["attention_mask"].to(args.device)

        labels = batch["label"].long().to(args.device)
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(args.device)
            output = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
            )
        else:
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=labels)

        res = torch.softmax(output.logits, axis=1)
        preds = res.argmax(1)
        denom += len(preds)

        nn_count += (torch.sum(res, axis=0)[1]).item()
        fn_count += (torch.bincount(preds)[1]).item()
        tn_count += torch.sum(res[:, 1] > args.tau).item()
        tn2_count += torch.sum(res[:, 1] > args.tau2).item()

        if batch_idx % args.print_every == 0:
            logger.info(f"net neutral: {nn_count / denom}")
            logger.info(f"fraction neutral: {fn_count / denom}")
            logger.info(f"tau 0.5 neutral: {tn_count / denom}")
            logger.info(f"tau 0.7 neutral: {tn2_count / denom}")

    logger.info(f"total net neutral: {nn_count / denom}")
    logger.info(f"total fraction neutral: {fn_count / denom}")
    logger.info(f"total tau 0.5 neutral: {tn_count / denom}")
    logger.info(f"total tau 0.7 neutral: {tn2_count / denom}")

    results = {"model_path": model_path,
               "total net neutral": nn_count / denom,
               "total fraction neutral": fn_count / denom,
               "total tau 0.5 neutral": tn_count / denom,
               "total tau 0.7 neutral": tn2_count / denom}

    # with open(os.path.join(output_dir, "bias_results_bias_nli.json"), "a") as f:
    #     json.dump(results, f)
    #     f.write('\n')
    with open(os.path.join(output_dir, "bias_results_bias_nli_whole.json"), "a") as f:
        json.dump(results, f)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau2", type=float, default=0.7)
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--eval_data_path", type=str, default="data/bias-nli/nli-dataset.csv")
    # parser.add_argument(
    #     "--load_from_file",
    #     type=str,
    #     default=None,
    #     help="path to evaluation .pt checkpoint",
    # )

    args = parser.parse_args()
    # args.update_encoder = True
    if os.path.basename(args.model_name_or_path).startswith('epoch'):
        previous_dir_path = os.path.dirname(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(previous_dir_path, use_fast=True)
        config = AutoConfig.from_pretrained(previous_dir_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        config = AutoConfig.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples["premise"],)
            if "hypothesis" == None
            else (examples["premise"], examples["hypothesis"])
        )
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
        return result

    eval_dataset = load_dataset(
        "csv",
        data_files=args.eval_data_path,
        # split="train[:-10%]",
        split="train",
        cache_dir=args.cache_dir,
    )
    eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=False
    )

    logger.info(f"Number of examples: {len(eval_dataset)}")

    eval_dataset.set_format('torch',['input_ids','attention_mask','token_type_ids','label'],output_all_columns=True)

    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=args.batch_size, shuffle=True
    )


    if os.path.basename(args.model_name_or_path).startswith('epoch'):
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config).cuda()
        model.eval()
        evaluate_bias_nli(model, args, eval_loader, args.model_name_or_path, args.model_name_or_path)
    else:
        for dir_name in os.listdir(args.model_name_or_path):
            dir_path = os.path.join(args.model_name_or_path, dir_name)
            if os.path.isdir(dir_path) and os.path.basename(dir_path).startswith('epoch'):
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
                config = AutoConfig.from_pretrained(args.model_name_or_path)
                model = BertForSequenceClassification.from_pretrained(dir_path, config=config).cuda()
                model.eval()
                evaluate_bias_nli(model, args, eval_loader, args.model_name_or_path, dir_path)


if __name__ == "__main__":
    main()