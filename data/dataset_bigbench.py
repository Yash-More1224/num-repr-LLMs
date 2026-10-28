"""
data/dataset_bigbench.py
=========================
PyTorch Dataset for training on the BigBench Arithmetic task.

The BigBench Arithmetic dataset has a very different format from ToTTo:
  - INPUT : "What is 3 + 4?"  (a natural language arithmetic question)
  - TARGET: "7"               (a single number, the answer)

Training strategy (causal LM, same as ToTTo):
  - Sequence: [INPUT] [SEP] [TARGET] [EOS]
  - Labels   : -100 for the input portion; target tokens after [SEP]

For xVal/FoNE: numbers in both input AND target are replaced with [NUM].
For digit    : numbers are split into digits.
For BPE      : numbers left as-is (standard BPE handles them).

All four model types share this same file; the preprocessing is controlled
by the `model_type` argument.

Usage
-----
  from data.dataset_bigbench import BigBenchArithmeticDataset, bigbench_collate_fn
  from tokenizers import Tokenizer

  tokenizer = Tokenizer.from_file("tokenizer/totto_bpe/tokenizer.json")
  dataset = BigBenchArithmeticDataset(
      data_path  = "data/bigbench_arithmetic/train.jsonl",
      tokenizer  = tokenizer,
      model_type = "bpe",   # one of: "bpe", "xval", "fone", "digit"
      max_seq_len = 128,
  )
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from data.dataset_xval import NUMBER_RE, parse_number, normalize_number, replace_numbers_with_num_token
from data.dataset_fone import replace_numbers_fone


class BigBenchArithmeticDataset(Dataset):
    """
    Dataset for BigBench Arithmetic training / evaluation.

    Handles all four model types with a single class to keep code DRY.

    Parameters
    ----------
    data_path   : str   — path to train.jsonl / dev.jsonl / test.jsonl
    tokenizer   : tokenizers.Tokenizer
    model_type  : str   — "bpe", "xval", "fone", or "digit"
    max_seq_len : int   — pad/truncate to this length
    """

    SUPPORTED_TYPES = {"bpe", "xval", "fone", "digit"}

    def __init__(self,
                 data_path:   str,
                 tokenizer:   Tokenizer,
                 model_type:  str = "bpe",
                 max_seq_len: int = 128):
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_TYPES}, got '{model_type}'")

        self.tokenizer   = tokenizer
        self.model_type  = model_type
        self.max_seq_len = max_seq_len

        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["[PAD]"]
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]
        self.num_id = vocab.get("[NUM]", None)

        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[BigBenchArithmetic/{model_type}] Loaded {len(self.examples):,} "
              f"examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def _preprocess_text(self, text: str):
        """
        Apply model-specific text preprocessing.

        Returns
        -------
        processed_text : str   — text after number substitution
        num_values     : list  — numeric values at [NUM] positions ([] for bpe/digit)
        """
        if self.model_type == "bpe":
            return text, []
        elif self.model_type == "digit":
            # Import here to avoid circular imports
            from tokenizer.train_digit_tokenizer import split_numbers_to_digits
            return split_numbers_to_digits(text), []
        elif self.model_type == "xval":
            proc_text, num_vals = replace_numbers_with_num_token(text)
            return proc_text, num_vals   # log-normalized
        elif self.model_type == "fone":
            proc_text, num_vals = replace_numbers_fone(text)
            return proc_text, num_vals   # raw values

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          input_ids      : LongTensor  (max_seq_len,)
          labels         : LongTensor  (max_seq_len,)  — -100 for input+padding
          attention_mask : LongTensor  (max_seq_len,)
          number_values  : FloatTensor (max_seq_len,)  — zeros for bpe/digit
          number_mask    : BoolTensor  (max_seq_len,)  — all-False for bpe/digit
          example_id     : str
          target         : str
        """
        record = self.examples[idx]

        # Build the full sequence: "<input> [SEP] <target> [EOS]"
        full_text = f"{record['input']} [SEP] {record['target']} [EOS]"

        # Apply model-specific preprocessing
        proc_text, num_values_ordered = self._preprocess_text(full_text)

        # Tokenize
        encoding  = self.tokenizer.encode(proc_text)
        token_ids = encoding.ids

        # Find [SEP] position to know where to start computing labels
        try:
            sep_pos = token_ids.index(self.sep_id)
        except ValueError:
            sep_pos = -1

        # Truncate to max_seq_len
        token_ids = token_ids[:self.max_seq_len]
        seq_len   = len(token_ids)

        # Build number_values and number_mask (only meaningful for xval/fone)
        num_iter  = iter(num_values_ordered)
        num_vals  = []
        num_mask  = []
        for tid in token_ids:
            if self.num_id is not None and tid == self.num_id:
                val = next(num_iter, 0.0)
                num_vals.append(float(val))
                num_mask.append(True)
            else:
                num_vals.append(0.0)
                num_mask.append(False)

        # Build labels: -100 for input portion, next-token IDs for target portion
        labels = []
        for i in range(seq_len):
            if i < sep_pos:
                labels.append(-100)
            elif i == seq_len - 1:
                labels.append(-100)
            else:
                labels.append(token_ids[i + 1])

        # Pad to max_seq_len
        pad_len        = self.max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids      = token_ids + [self.pad_id] * pad_len
        labels         = labels    + [-100]        * pad_len
        num_vals       = num_vals  + [0.0]         * pad_len
        num_mask       = num_mask  + [False]       * pad_len

        return {
            "input_ids":      torch.tensor(token_ids,      dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "number_values":  torch.tensor(num_vals,       dtype=torch.float32),
            "number_mask":    torch.tensor(num_mask,       dtype=torch.bool),
            "example_id":     record["id"],
            "target":         record["target"],
        }


def bigbench_collate_fn(batch: list) -> dict:
    """Collate BigBench samples — same structure for all model types."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "number_values":  torch.stack([b["number_values"]  for b in batch]),
        "number_mask":    torch.stack([b["number_mask"]    for b in batch]),
        "example_ids":    [b["example_id"] for b in batch],
        "targets":        [b["target"]     for b in batch],
    }
