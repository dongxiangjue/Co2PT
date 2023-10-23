import json
import logging
import os
import random
from time import time
import torch.distributed
from pathlib import Path
import argparse
import pickle

import yaml
import glob
import logging
from datasets import Dataset, load_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    set_seed,
)
from promptBERTSeq import BertPrefixForSequenceClassification

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)

with open("data/biasbios/prof2ind.json") as json_file:
    mapping = json.load(json_file)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def align_loss_fct(x, y, alpha=2):  # from Wang and Isola, 2018
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def sim(x, y, temp):
    cos = nn.CosineSimilarity(dim=-1)
    return cos(x, y) / temp


def save_ckpt(model, optimizer, args, latest=False):
    if not latest:
        best_ckpt_path = args.file_path.format(
            step=args.global_step, best_score=args.best_score * 100
        )
        checkpoint = {"ckpt_path": best_ckpt_path}
    else:
        checkpoint = {"ckpt_path": os.path.join(args.ckpt_dir, "latest.ckpt")}

    states = model.state_dict() if not args.dataparallel else model.module.state_dict()
    checkpoint["states"] = states
    checkpoint["optimizer_states"] = optimizer.state_dict()

    if not latest:
        for rm_path in glob.glob(os.path.join(args.ckpt_dir, "*.pt")):
            os.remove(rm_path)

    torch.save(checkpoint, checkpoint["ckpt_path"])
    print(f"Model saved at: {checkpoint['ckpt_path']}")


def save_args(args):
    with open(args.args_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Arg file saved at: {args.args_path}")


def process_data(dataset):
    occ = []
    bio = []
    gend = []

    for elem in dataset:
        occ.append(elem["p"])
        bio.append(elem["hard_text_untokenized"])
        gend.append(elem["g"])

    prof_result = []

    for _, v in enumerate(occ):
        try:
            index = mapping[v]
        except KeyError:
            raise Exception("unknown label in occupation")
        prof_result.append(index)

    gend_result = []

    for _, v in enumerate(gend):
        if v == "m":
            gend_result.append(0)
        elif v == "f":
            gend_result.append(1)
        else:
            raise Exception("unknown label in gender")

    data_dict = {"label": prof_result, "bio": bio, "gend": gend_result}
    dataset = Dataset.from_dict(data_dict)
    return dataset


def train_epoch(epoch, model, dl, eval_dl, aug_dl, optimizer, args):
    logger.info(f"At epoch {epoch}:")
    model.train()

    t = time()
    total_loss = 0
    total_ft_loss = 0
    total_a_loss = 0
    total_correct = []

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        labels = torch.tensor(batch["label"]).long().to(args.device)

        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        ft_loss = output.loss
        logits = output.logits

        item = next(iter(aug_dl))
        # orig_z0_input_ids, orig_z0_mask, orig_z0_token_ids = item["input_ids"][:, 0, :].to(args.device), item["attention_mask"][:, 0, :].to(args.device), item["token_type_ids"][:, 0, :].to(args.device)
        # aug_z0_input_ids, aug_z0_mask, aug_z0_token_ids = item["input_ids"][:, 1, :].to(args.device), item["attention_mask"][:, 1, :].to(args.device), item["token_type_ids"][:, 1, :].to(args.device)
        orig_z0_input_ids = torch.transpose(torch.stack(item["input_ids"][0]), 0, 1).to(
            args.device
        )
        orig_z0_mask = torch.transpose(torch.stack(item["attention_mask"][0]), 0, 1).to(
            args.device
        )
        aug_z0_input_ids = torch.transpose(torch.stack(item["input_ids"][1]), 0, 1).to(
            args.device
        )
        aug_z0_mask = torch.transpose(torch.stack(item["attention_mask"][1]), 0, 1).to(
            args.device
        )

        orig_z0_outputs = model(
            input_ids=orig_z0_input_ids,
            attention_mask=orig_z0_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]
        aug_z0_outputs = model(
            input_ids=aug_z0_input_ids,
            attention_mask=aug_z0_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]

        # last_hidden_states = outputs["hidden_states"][-1]
        # enc = model.bert.pooler(last_hidden_states)

        orig_z0 = model.bert.pooler(orig_z0_outputs)
        aug_z0 = model.bert.pooler(aug_z0_outputs)

        if args.cl_loss:
            cos_sim = sim(orig_z0.unsqueeze(1), aug_z0.unsqueeze(0), temp=args.temp)
            cl_labels = torch.arange(cos_sim.size(0)).long().to(args.device)
            loss_fct = nn.CrossEntropyLoss()
            a_loss = loss_fct(cos_sim, cl_labels)
        else:
            a_loss = align_loss_fct(orig_z0, aug_z0)

        loss = ft_loss + a_loss * args.align_temp

        if args.dataparallel:
            total_loss += loss.mean().item()
            total_ft_loss += ft_loss.mean().item()
            total_a_loss += a_loss.mean().item()
            loss.mean().backward()
        else:
            total_loss += loss.item()
            total_ft_loss += ft_loss.item()
            total_a_loss += a_loss.item()
            loss.backward()

        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()

        if args.global_step % args.print_every == 0:
            train_acc = sum(total_correct) / len(total_correct)
            train_loss = total_loss / len(total_correct)
            train_ft_loss = total_ft_loss / len(total_correct)
            train_a_loss = total_a_loss / len(total_correct)
            logger.info(f"train_acc: {train_acc}, train_total_loss: {train_loss}, train_ft_loss: {train_ft_loss}, train_a_loss: {train_a_loss}")

        if args.global_step % args.eval_interval == 0 and args.global_step != 0:
            dev_acc = evaluate("dev", model, eval_dl, epoch, args)
            logger.info(f"dev_acc: {dev_acc}")
            if dev_acc > args.best_score:
                args.best_score = dev_acc
                logger.info(f"best_acc: {args.best_score}")
            save_ckpt(model, optimizer, args, latest=False)

        args.global_step += 1


def evaluate(mode, model, dl, epoch, args):
    model.eval()
    logger.info("running eval")

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        labels = torch.tensor(batch["label"]).long().to(args.device)

        batch_loss = None
        logits = None

        with torch.no_grad():
            if args.dataparallel:
                output = model.module.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            else:
                output = model.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

        batch_loss = output.loss
        logits = output.logits

        if args.dataparallel:
            total_loss += batch_loss.sum().item()
        else:
            total_loss += batch_loss.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    log_dict = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "eval_acc": acc,
        "eval_loss": loss,
        "global_step": args.global_step,
    }
    logger.info(mode, log_dict)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=False)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--print_every", default=50, type=int)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--ckpt_dir", default="0519_test", type=str)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument(
        "--file_name", default="model-step={step}-acc={best_score:.2f}.pt"
    )
    parser.add_argument("--train_path", type=str, default="data/biasbios/train.pickle")
    parser.add_argument("--val_path", type=str, default="data/biasbios/val.pickle")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument(
        "--max_seq_len", default=128, type=int, help="Max. sequence length"
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--fix_encoder",
        action="store_true",
        help="Whether to fix encoder - default false.",
    )
    parser.add_argument("--pre_seq_len", default=5, type=int, help="The length of prompt.")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="Apply a two-layer MLP head over the prefix embeddings.")
    parser.add_argument("--prefix_hidden_size", default=512, type=int, help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="The dropout probability used in the models.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--align_temp", type=float, default=0.05, help="parameter before align loss")
    parser.add_argument("--cl_loss", action="store_true", help="use the contrastive loss.")
    parser.add_argument("--temp", type=float, default=0.05, help="parameter for cosine similarity")

    args = parser.parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    args.best_score = float("-inf")
    args.global_step = 0
    args.args_path = os.path.join(args.ckpt_dir, "args.yaml")
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)
    save_args(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    train_file = open(args.train_path, "rb")
    train_data = pickle.load(train_file)
    train_file.close()

    val_file = open(args.train_path, "rb")
    val_data = pickle.load(val_file)
    val_file.close()

    raw_datasets2 = load_dataset(
        "csv",
        data_files=f"data/train_aug/biasbios-train-gender.csv",
        delimiter=",",
    )

    train_dataset = process_data(train_data)
    val_dataset = process_data(val_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )

    def preprocess_function(examples):
        # Tokenize the texts
        args = (examples["bio"],)
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
        return result

    train_dataset = train_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=False
    )
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=False
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(mapping),
        output_hidden_states=True,
    )
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection
    config.prefix_hidden_size = args.prefix_hidden_size
    model = BertPrefixForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    logger.info(f"load model from {args.model_name_or_path}.")


    if args.fix_encoder:
        print("FIXING ENCODER...")
        if "roberta" in args.model_name_or_path:
            for param in model.roberta.parameters():
                param.requires_grad = False
        else:
            for param in model.bert.parameters():
                param.requires_grad = False

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.warning(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False
    )
    eval_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )

    raw_datasets2 = load_dataset(
        "csv",
        data_files=f"data/train_aug/biasbios-train-gender.csv",
        delimiter=",",
    )

    # Prepare features
    column_names = raw_datasets2["train"].column_names
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]


    def preprocess_function2(examples):
        # sent0_cname = 'orig_sent0'
        # sent1_cname = 'orig_sent1'
        # sent2_cname = 'aug_sent0'
        # sent3_cname = 'aug_sent1'

        total = len(examples[sent0_cname])
        # p, p', h, h'
        orisent_features = tokenizer(
            examples[sent0_cname],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        augsent_features = tokenizer(
            examples[sent1_cname],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        features = {}

        for key in orisent_features:
            features[key] = [
                [
                    orisent_features[key][i],
                    augsent_features[key][i],
                ]
                for i in range(total)
            ]

        return features

    processed_datasets2 = raw_datasets2.map(
        preprocess_function2,
        batched=True,
        remove_columns=raw_datasets2["train"].column_names,
        desc="Running tokenizer on dataset",
        load_from_cache_file=False
    )

    train_dataset2 = processed_datasets2["train"]
    train_dataloader2 = DataLoader(train_dataset2, shuffle=True, batch_size=args.batch_size)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # Log a few random samples from the cls set:
    for index in random.sample(range(len(train_dataset2)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset2[index]}.")

    if args.dataparallel:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}"
        )
        logger.info(
            f"All params      : {sum(p.numel() for p in model.module.parameters())}"
        )
    else:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        logger.info(f"All params      : {sum(p.numel() for p in model.parameters())}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    for epoch in range(int(args.num_epochs)):
        train_epoch(epoch, model, train_loader, eval_loader, train_dataloader2, optimizer, args)

    save_ckpt(model, optimizer, args, latest=True)
    logger.info(f"Finished training with highest dev acc. {args.best_score}")


if __name__ == "__main__":
    main()