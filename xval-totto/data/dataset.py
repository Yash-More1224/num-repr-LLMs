"""
data/dataset.py
===============
PyTorch Dataset for the BPE-GPT2 (baseline) model.

What this does
--------------
Reads the processed train.jsonl or dev.jsonl (produced by preprocess.py),
tokenizes each linearized string using the trained BPE tokenizer, and
returns (input_ids, labels, attention_mask) tensors ready for the model.

Loss masking
------------
The model is trained to predict only the TARGET tokens (after [SEP]).
Tokens in the input portion (before and including [SEP]) are masked with -100
so the cross-entropy loss ignores them.

This makes the task equivalent to seq2seq: the model learns to generate
a faithful sentence conditioned on the linearized table input.

Tensor shapes (all fixed at max_seq_len)
-----------------------------------------
  input_ids      : (max_seq_len,)   int64 — token IDs, padded with PAD_ID
  labels         : (max_seq_len,)   int64 — same as input_ids but -100 where
                                           we do NOT want loss computed
  attention_mask : (max_seq_len,)   int64 — 1 for real tokens, 0 for padding

Usage
-----
  from data.dataset import ToTToDataset
  from tokenizers import Tokenizer

  tokenizer = Tokenizer.from_file("tokenizer/totto_bpe/tokenizer.json")
  dataset = ToTToDataset("processed/train.jsonl", tokenizer, max_seq_len=256)
  sample = dataset[0]
  print(sample["input_ids"].shape)   # torch.Size([256])
"""

import json
import math
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class ToTToDataset(Dataset):
    """
    Dataset for the baseline BPE-GPT2 model.

    Parameters
    ----------
    data_path    : str  Path to processed/train.jsonl or processed/dev.jsonl
    tokenizer    : tokenizers.Tokenizer  The trained BPE tokenizer
    max_seq_len  : int  Maximum sequence length (pad/truncate to this)
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        # Read special token IDs from the tokenizer vocabulary
        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["[PAD]"]
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]

        # Load all examples into memory.
        # ToTTo has 120K train examples — each linearized string is ~200 chars.
        # Total memory: ~120K * 200 bytes ≈ 24 MB. Fine to keep in RAM.
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[ToTToDataset] Loaded {len(self.examples):,} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          input_ids      : LongTensor (max_seq_len,)
          labels         : LongTensor (max_seq_len,)
          attention_mask : LongTensor (max_seq_len,)
          example_id     : str (for debugging / evaluation matching)
          target         : str (reference sentence, for evaluation only)
        """
        record = self.examples[idx]
        linearized = record["linearized"]

        # ── Tokenize the full linearized string ───────────────────────────────
        encoding = self.tokenizer.encode(linearized)
        token_ids = encoding.ids   # list of int

        # ── Find the position of [SEP] in the token sequence ─────────────────
        # Everything from position 0 up to and including sep_pos is the INPUT.
        # Everything after sep_pos is the TARGET (where we compute loss).
        try:
            sep_pos = token_ids.index(self.sep_id)
        except ValueError:
            # Fallback: if [SEP] is not found (shouldn't happen), treat
            # the whole sequence as target.
            sep_pos = -1

        # ── Truncate to max_seq_len ───────────────────────────────────────────
        # If the sequence is too long, truncate from the END (keep the input
        # and as much of the target as fits).
        token_ids = token_ids[: self.max_seq_len]

        seq_len = len(token_ids)

        # ── Build labels ──────────────────────────────────────────────────────
        # Labels mirror input_ids but shifted by 1 (next-token prediction objective).
        #
        # In a causal GPT transformer:
        #   logits[i] = P(next_token | tokens_0..i)
        #
        # So logits[i] should be trained against token_ids[i+1], NOT token_ids[i].
        # Using token_ids[i] (unshifted) would let the model trivially "copy" the
        # current token it already sees at position i, collapsing training loss to ~0
        # without learning anything generative.
        #
        # Masking boundary is i < sep_pos (strict less-than), NOT i <= sep_pos:
        #   - Positions 0 .. sep_pos-1 : logits predict INPUT tokens → mask (-100)
        #   - Position sep_pos ([SEP]) : logits predict FIRST target token → TRAIN
        #   - Positions sep_pos+1 ..  : logits predict subsequent target tokens → TRAIN
        #   - Last position (seq_len-1): nothing to predict → mask (-100)
        labels = []
        for i in range(seq_len):
            if i < sep_pos:
                # This position predicts an input token → ignore in loss
                labels.append(-100)
            elif i == seq_len - 1:
                # Last position — no next token to predict → ignore
                labels.append(-100)
            else:
                # This position predicts a target token (or first target from [SEP])
                labels.append(token_ids[i + 1])

        # ── Pad to max_seq_len ────────────────────────────────────────────────
        pad_length = self.max_seq_len - seq_len
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


# ── Collate function for DataLoader ──────────────────────────────────────────
# The default PyTorch collate_fn can't handle the string fields (example_id,
# target). This custom collator handles it cleanly.

def bpe_collate_fn(batch: list) -> dict:
    """
    Collate a list of dataset samples into a batch.

    Stacks all tensor fields and keeps string fields as lists.
    """
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "example_ids":    [b["example_id"] for b in batch],
        "targets":        [b["target"]     for b in batch],
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this directly to verify the dataset loads correctly:
        python data/dataset.py
    """
    import sys
    from torch.utils.data import DataLoader

    # Paths — adjust if running from a different directory
    tokenizer_path = "tokenizer/totto_bpe/tokenizer.json"
    data_path      = "processed/train.jsonl"

    print("Loading tokenizer ...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print("Loading dataset ...")
    dataset = ToTToDataset(data_path, tokenizer, max_seq_len=256)

    print(f"\nDataset size: {len(dataset):,}")

    # Check a few samples
    for i in [0, 1, 2]:
        sample = dataset[i]
        print(f"\n── Sample {i} ──────────────────────────────")
        print(f"  target         : {sample['target'][:80]}")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  n_real_tokens  : {sample['attention_mask'].sum().item()}")
        # Count how many positions have real labels (target portion)
        n_target_tokens = (sample["labels"] != -100).sum().item()
        print(f"  n_target_tokens: {n_target_tokens}")
        # Decode input and target portions separately
        sep_pos = (sample["input_ids"] == tokenizer.get_vocab()["[SEP]"]).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            input_tokens  = sample["input_ids"][:sep_pos[0]+1].tolist()
            target_tokens = sample["input_ids"][sep_pos[0]+1:].tolist()
            target_tokens = [t for t in target_tokens if t != tokenizer.get_vocab()["[PAD]"]]
            print(f"  input decoded  : {tokenizer.decode(input_tokens)[:80]}")
            print(f"  target decoded : {tokenizer.decode(target_tokens)[:80]}")

    # Test the DataLoader
    print("\n── DataLoader test ─────────────────────────────")
    loader = DataLoader(dataset, batch_size=4, collate_fn=bpe_collate_fn, shuffle=False)
    batch = next(iter(loader))
    print(f"  input_ids shape      : {batch['input_ids'].shape}")
    print(f"  labels shape         : {batch['labels'].shape}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}")
    print(f"  targets (first 2)    : {batch['targets'][:2]}")
    print("\n✓ BPE Dataset OK")
