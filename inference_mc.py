import argparse
import json
import os
from dataclasses import dataclass
from itertools import chain
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorWithPadding
)
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--context_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing context data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    contexts = None
    if args.context_file is not None:
        with open(args.context_file ,encoding="utf-8") as f:
                contexts = json.load(f)

    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["question"]]   
        second_sentences = [
            [f"{contexts[p]}" for p in p_list] for p_list in examples["paragraphs"]
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=True,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        return tokenized_inputs

    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer)
    
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
    )
    test_dataloader = DataLoader(processed_datasets["test"], batch_size=8, collate_fn=data_collator)

    model.to("cuda")
    model.eval()
    
    all_predictions = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to("cuda")
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
    # print(all_predictions)
    

    ids = raw_datasets["test"]["id"]
    results = [{"id": id_, "answer": int(pred)} for id_, pred in zip(ids, all_predictions)]
    output_file = os.path.join("mc_pred.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {output_file}")
    

if __name__ == "__main__":
    main()