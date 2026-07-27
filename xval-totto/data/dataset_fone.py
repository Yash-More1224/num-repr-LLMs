"""
data/dataset_fone.py
====================
PyTorch Dataset for the FoNE-GPT2 model.

Identical in structure to dataset_xval.py, with TWO key differences:

  1. number_values contains RAW (unnormalized) float values, not log-normalized.
     FoNE uses raw values because the Fourier encoding operates directly on
     the number magnitude — log-normalization would distort the periodicity.

  2. No MSE regression target is needed — FoNE uses CE loss only. The
     number_values tensor is used ONLY by the FoNEEncoder.encode_sequence()
     to build the additive Fourier embedding for the input side.

The returned tensors are otherwise identical to dataset_xval.py so that the
same Trainer, collate_fn, and training loop can be reused.

Tensor shapes (all fixed at max_seq_len)
-----------------------------------------
  input_ids      : (max_seq_len,)   int64   — token IDs ([NUM] for numbers)
  labels         : (max_seq_len,)   int64   — -100 for input + padding
  attention_mask : (max_seq_len,)   int64   — 1 for real tokens
  number_values  : (max_seq_len,)   float32 — RAW value at [NUM] positions
  number_mask    : (max_seq_len,)   bool    — True at [NUM] positions
"""

import json
import re
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Re-use the number detection regex and parse_number from dataset_xval
from data.dataset_xval import NUMBER_RE, parse_number


def replace_numbers_fone(text: str) -> tuple:
    """
    Find all number spans in `text`, replace each with '[NUM]', and return
    the modified string along with the ordered list of RAW number values.

    Unlike the xVal version, we store raw values (not log-normalized),
    because FoNE's Fourier encoding operates on the actual magnitude.

    Parameters
    ----------
    text : str

    Returns
    -------
    fone_text     : str   — text with all numbers replaced by '[NUM]'
    number_values : list  — [float, ...] RAW values, in left-to-right order
    """
    number_values = []
    result_parts  = []
    last_end      = 0

    for match in NUMBER_RE.finditer(text):
        start, end = match.start(), match.end()
        num_str    = match.group(1)
        try:
            value = parse_number(num_str)
        except ValueError:
            continue
        result_parts.append(text[last_end:start])
        result_parts.append("[NUM]")
        number_values.append(value)   # RAW value (no log normalization)
        last_end = end

    result_parts.append(text[last_end:])
    fone_text = "".join(result_parts)
    return fone_text, number_values


class ToTToDatasetFoNE(Dataset):
    """
    Dataset for the FoNE-GPT2 model.

    Same as ToTToDatasetXVal but stores RAW number values (not log-normalized)
    since FoNE Fourier encoding needs the true magnitude.

    Parameters
    ----------
    data_path   : str
    tokenizer   : tokenizers.Tokenizer
    max_seq_len : int
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["[PAD]"]
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]
        self.num_id = vocab["[NUM]"]

        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[ToTToDatasetFoNE] Loaded {len(self.examples):,} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        record     = self.examples[idx]
        linearized = record["linearized"]

        # Replace numbers with [NUM], keep raw values
        fone_text, raw_values = replace_numbers_fone(linearized)

        # Tokenize
        encoding  = self.tokenizer.encode(fone_text)
        token_ids = encoding.ids

        # Find [SEP] position
        try:
            sep_pos = token_ids.index(self.sep_id)
        except ValueError:
            sep_pos = -1

        # Truncate
        token_ids = token_ids[:self.max_seq_len]
        seq_len   = len(token_ids)

        # Build number_values and number_mask aligned to token positions
        # Walk the token list; each [NUM] token consumes one value from raw_values
        num_iter  = iter(raw_values)
        num_vals  = []
        num_mask  = []
        for tid in token_ids:
            if tid == self.num_id:
                val = next(num_iter, 0.0)
                num_vals.append(float(val))
                num_mask.append(True)
            else:
                num_vals.append(0.0)
                num_mask.append(False)

        # Build labels (next-token prediction, mask input portion and last pos)
        labels = []
        for i in range(seq_len):
            if i < sep_pos:
                labels.append(-100)
            elif i == seq_len - 1:
                labels.append(-100)
            else:
                labels.append(token_ids[i + 1])

        # Pad to max_seq_len
        pad_length     = self.max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0] * pad_length
        token_ids      = token_ids + [self.pad_id] * pad_length
        labels         = labels    + [-100]        * pad_length
        num_vals       = num_vals  + [0.0]         * pad_length
        num_mask       = num_mask  + [False]       * pad_length

        return {
            "input_ids":      torch.tensor(token_ids,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "number_values":  torch.tensor(num_vals,   dtype=torch.float32),
            "number_mask":    torch.tensor(num_mask,   dtype=torch.bool),
            "example_id":     record["example_id"],
            "target":         record["target"],
        }


def fone_collate_fn(batch: list) -> dict:
    """Collate FoNE samples into a batch."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "number_values":  torch.stack([b["number_values"]  for b in batch]),
        "number_mask":    torch.stack([b["number_mask"]    for b in batch]),
        "example_ids":    [b["example_id"] for b in batch],
        "targets":        [b["target"]     for b in batch],
    }
