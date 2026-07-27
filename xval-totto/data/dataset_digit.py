"""
data/dataset_digit.py
======================
PyTorch Dataset for the Digit Tokenization model.

The ONLY difference from dataset.py (BPE baseline):
  - Before tokenizing, all number spans are pre-processed by splitting
    their digits with spaces: "1874" -> "1 8 7 4"
  - This works with the totto_digit tokenizer (trained by train_digit_tokenizer.py)
    which guarantees each digit 0-9 is a single token.

Everything else (loss masking, padding, collate_fn) is identical to the BPE dataset.

The digit tokenizer must be loaded from tokenizer/totto_digit/tokenizer.json.

Usage
-----
  from data.dataset_digit import ToTToDatasetDigit, digit_collate_fn
  from tokenizers import Tokenizer

  tokenizer = Tokenizer.from_file("tokenizer/totto_digit/tokenizer.json")
  dataset = ToTToDatasetDigit("processed/train.jsonl", tokenizer, max_seq_len=512)
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from tokenizer.train_digit_tokenizer import split_numbers_to_digits


class ToTToDatasetDigit(Dataset):
    """
    Dataset for the Digit Tokenization model.

    Parameters
    ----------
    data_path   : str
    tokenizer   : tokenizers.Tokenizer  — the digit tokenizer
    max_seq_len : int  — digit sequences are longer than BPE (more tokens per number)
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["[PAD]"]
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]

        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[ToTToDatasetDigit] Loaded {len(self.examples):,} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        record     = self.examples[idx]
        linearized = record["linearized"]

        # Pre-process: split number spans into space-separated digits
        # "The 76th Governor from 1874" -> "The 7 6 th Governor from 1 8 7 4"
        digit_text = split_numbers_to_digits(linearized)

        encoding  = self.tokenizer.encode(digit_text)
        token_ids = encoding.ids

        try:
            sep_pos = token_ids.index(self.sep_id)
        except ValueError:
            sep_pos = -1

        token_ids = token_ids[:self.max_seq_len]
        seq_len   = len(token_ids)

        # Labels: next-token prediction, mask input portion
        labels = []
        for i in range(seq_len):
            if i < sep_pos:
                labels.append(-100)
            elif i == seq_len - 1:
                labels.append(-100)
            else:
                labels.append(token_ids[i + 1])

        pad_length     = self.max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0] * pad_length
        token_ids      = token_ids + [self.pad_id] * pad_length
        labels         = labels    + [-100]        * pad_length

        return {
            "input_ids":      torch.tensor(token_ids,      dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "example_id":     record["example_id"],
            "target":         record["target"],
        }


def digit_collate_fn(batch: list) -> dict:
    """Collate digit samples into a batch."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "example_ids":    [b["example_id"] for b in batch],
        "targets":        [b["target"]     for b in batch],
    }
