"""
data/dataset_xval.py
====================
PyTorch Dataset for the xVal-GPT2 model.

The ONLY difference from dataset.py (BPE baseline):
  - Before tokenizing, all number spans are detected and replaced with [NUM].
  - Two extra tensors are returned per sample:
      number_values : float32 tensor — the actual numeric value at each [NUM]
                      position, 0.0 everywhere else.
      number_mask   : bool tensor — True at every [NUM] token position,
                      False everywhere else.

These tensors are consumed by the xVal model to:
  (a) Scale the [NUM] embedding by the actual number value (input side).
  (b) Compute an MSE regression loss on number positions (output side).

Number detection
----------------
We use a regex that handles:
  - Integers            : 42, 2001, 1943
  - Decimals            : 44.9, 3.14, 0.089
  - Comma-grouped ints  : 281,893  -> 281893.0
  - Large values        : 3,141,592.65 -> 3141592.65

Number normalization
--------------------
Raw number values can range from 0 to millions. We log-normalize them before
storing, so the regression loss operates in a stable range:
  normalized = sign(x) * log1p(|x|)
The model's scalar head predicts in this normalized space.
The inverse transform recovers the original value for evaluation.

Tensor shapes (all fixed at max_seq_len)
-----------------------------------------
  input_ids      : (max_seq_len,)   int64   — token IDs ([NUM] for numbers)
  labels         : (max_seq_len,)   int64   — -100 for input + padding
  attention_mask : (max_seq_len,)   int64   — 1 for real tokens
  number_values  : (max_seq_len,)   float32 — normalized value at [NUM] pos
  number_mask    : (max_seq_len,)   bool    — True at [NUM] positions

Usage
-----
  from data.dataset_xval import ToTToDatasetXVal
  from tokenizers import Tokenizer

  tokenizer = Tokenizer.from_file("tokenizer/totto_bpe/tokenizer.json")
  dataset = ToTToDatasetXVal("processed/train.jsonl", tokenizer, max_seq_len=256)
  sample = dataset[0]
  print(sample["number_mask"].sum())   # number of [NUM] tokens in this sample
"""

import json
import math
import re
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# ── Number detection regex ────────────────────────────────────────────────────
# Matches:
#   - comma-grouped integers: 1,234,567
#   - decimal numbers:        44.9, 3.14
#   - comma-grouped decimals: 1,234.56
#   - plain integers:         42, 2001
#
# Does NOT match:
#   - Special tokens like [TITLE], [SEP], [NUM] (they don't contain digits
#     surrounded by word boundaries in this way)
#   - Ordinals like "1st", "2nd" — the \b ensures the digit run is isolated
#
# Group 1 captures the full number string (with commas if present).
NUMBER_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b"
)


def parse_number(num_str: str) -> float:
    """Parse a number string (possibly with commas) into a float."""
    return float(num_str.replace(",", ""))


def normalize_number(x: float) -> float:
    """
    Log-normalize a number value for stable regression training.

    Maps any real number to a compact range:
      0           -> 0.0
      1           -> log(2) ≈ 0.693
      100         -> log(101) ≈ 4.615
      1,000,000   -> log(1_000_001) ≈ 13.816
      -44.9       -> -log(45.9) ≈ -3.826

    Use denormalize_number() to recover the original value.
    """
    if x == 0.0:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    return sign * math.log1p(abs(x))


def denormalize_number(x: float) -> float:
    """Inverse of normalize_number — recover original value from log-space."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0 else -1.0
    return sign * math.expm1(abs(x))


def replace_numbers_with_num_token(text: str) -> tuple:
    """
    Find all number spans in `text`, replace each with '[NUM]', and return
    the modified string along with the ordered list of number values.

    Parameters
    ----------
    text : str  The linearized string (may contain numbers anywhere)

    Returns
    -------
    xval_text     : str    The text with all numbers replaced by '[NUM]'
    number_values : list   [float, ...] — log-normalized value for each [NUM],
                           in left-to-right order of appearance in text.

    Example
    -------
    Input : "[HL] Acres: 281,893 [SEP] Thomas burned 281,893 acres. [EOS]"
    Output: ("[HL] Acres: [NUM] [SEP] Thomas burned [NUM] acres. [EOS]",
             [12.548, 12.548])   # log1p(281893) ≈ 12.548
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
            # Can't parse — leave this span as-is (don't replace)
            continue

        # Append text before this number
        result_parts.append(text[last_end:start])
        # Append [NUM] placeholder
        result_parts.append("[NUM]")
        # Store the normalized value
        number_values.append(normalize_number(value))
        last_end = end

    # Append remaining text after the last number
    result_parts.append(text[last_end:])

    xval_text = "".join(result_parts)
    return xval_text, number_values


class ToTToDatasetXVal(Dataset):
    """
    Dataset for the xVal-GPT2 model.

    Identical to ToTToDataset (BPE baseline) except:
      1. Numbers in the linearized string are replaced with [NUM] before
         tokenization.
      2. Extra tensors number_values and number_mask are returned.

    Parameters
    ----------
    data_path    : str  Path to processed/train.jsonl or processed/dev.jsonl
    tokenizer    : tokenizers.Tokenizer  The trained BPE tokenizer
    max_seq_len  : int  Maximum sequence length (pad/truncate to this)
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len

        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["[PAD]"]
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]
        self.num_id = vocab["[NUM]"]   # ← the key token ID for xVal

        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[ToTToDatasetXVal] Loaded {len(self.examples):,} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          input_ids      : LongTensor  (max_seq_len,)
          labels         : LongTensor  (max_seq_len,)
          attention_mask : LongTensor  (max_seq_len,)
          number_values  : FloatTensor (max_seq_len,) — log-normalized values
          number_mask    : BoolTensor  (max_seq_len,) — True at [NUM] positions
          example_id     : str
          target         : str
        """
        record     = self.examples[idx]
        linearized = record["linearized"]

        # ── Step 1: Replace number spans with [NUM] ───────────────────────────
        xval_text, num_values_ordered = replace_numbers_with_num_token(linearized)
        # num_values_ordered is a list of float values, one per [NUM] in order.

        # ── Step 2: Tokenize the modified string ──────────────────────────────
        encoding  = self.tokenizer.encode(xval_text)
        token_ids = encoding.ids   # list of int

        # ── Step 3: Locate [SEP] for loss masking ────────────────────────────
        try:
            sep_pos = token_ids.index(self.sep_id)
        except ValueError:
            sep_pos = -1

        # ── Step 4: Truncate ─────────────────────────────────────────────────
        token_ids = token_ids[: self.max_seq_len]
        seq_len   = len(token_ids)

        # ── Step 5: Build number_values and number_mask ───────────────────────
        # Find positions of [NUM] tokens in the (possibly truncated) sequence.
        # Assign values left-to-right from num_values_ordered.
        number_values = [0.0] * seq_len
        number_mask   = [False] * seq_len

        num_counter = 0  # index into num_values_ordered
        for pos, tok_id in enumerate(token_ids):
            if tok_id == self.num_id:
                if num_counter < len(num_values_ordered):
                    number_values[pos] = num_values_ordered[num_counter]
                    num_counter += 1
                number_mask[pos] = True
                # Even if we ran out of stored values (edge case from truncation),
                # mark the mask — the value stays 0.0 (treated as 0 during forward pass)

        # ── Step 6: Build labels ──────────────────────────────────────────────
        # Same fix as BPE dataset: labels are SHIFTED by 1 (next-token prediction).
        #
        # logits[i] = P(next_token | tokens_0..i), so it must be trained against
        # token_ids[i+1]. Using token_ids[i] (unshifted) trivially collapses loss.
        #
        # Masking boundary: i < sep_pos (strict), so the [SEP] position at sep_pos
        # correctly trains the model to predict the first target token (token_ids[sep_pos+1]).
        labels = []
        for i in range(seq_len):
            if i < sep_pos:
                # Predicts an input token → ignore
                labels.append(-100)
            elif i == seq_len - 1:
                # Last position → nothing to predict → ignore
                labels.append(-100)
            else:
                # Predicts a target token → train
                labels.append(token_ids[i + 1])

        # ── Step 7: Pad to max_seq_len ────────────────────────────────────────
        pad_length     = self.max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0]     * pad_length
        token_ids      = token_ids     + [self.pad_id] * pad_length
        labels         = labels        + [-100]   * pad_length
        number_values  = number_values + [0.0]    * pad_length
        number_mask    = number_mask   + [False]  * pad_length

        return {
            "input_ids":      torch.tensor(token_ids,     dtype=torch.long),
            "labels":         torch.tensor(labels,        dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,dtype=torch.long),
            "number_values":  torch.tensor(number_values, dtype=torch.float32),
            "number_mask":    torch.tensor(number_mask,   dtype=torch.bool),
            "example_id":     record["example_id"],
            "target":         record["target"],
        }


# ── Collate function for DataLoader ──────────────────────────────────────────

def xval_collate_fn(batch: list) -> dict:
    """Collate a list of xVal samples into a batch."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "number_values":  torch.stack([b["number_values"]  for b in batch]),
        "number_mask":    torch.stack([b["number_mask"]    for b in batch]),
        "example_ids":    [b["example_id"] for b in batch],
        "targets":        [b["target"]     for b in batch],
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this directly to verify xVal preprocessing is correct:
        python data/dataset_xval.py
    """
    from torch.utils.data import DataLoader

    tokenizer_path = "tokenizer/totto_bpe/tokenizer.json"
    data_path      = "processed/train.jsonl"

    print("Loading tokenizer ...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    num_id    = tokenizer.get_vocab()["[NUM]"]

    print("Loading dataset ...")
    dataset = ToTToDatasetXVal(data_path, tokenizer, max_seq_len=256)
    print(f"Dataset size: {len(dataset):,}\n")

    # ── Unit test: number replacement ────────────────────────────────────────
    test_cases = [
        ("Thomas burned 281,893 acres.",
         "Thomas burned [NUM] acres.",
         [normalize_number(281893.0)]),
        ("Barbour served until November 22, 1943.",
         "Barbour served until November [NUM], [NUM].",
         [normalize_number(22.0), normalize_number(1943.0)]),
        ("No numbers here.",
         "No numbers here.",
         []),
    ]
    print("── Number replacement unit tests ──────────────────────")
    all_passed = True
    for original, expected_text, expected_vals in test_cases:
        got_text, got_vals = replace_numbers_with_num_token(original)
        ok_text = got_text == expected_text
        ok_vals = (
            len(got_vals) == len(expected_vals)
            and all(abs(a - b) < 1e-6 for a, b in zip(got_vals, expected_vals))
        )
        status = "✓" if (ok_text and ok_vals) else "✗"
        print(f"  {status}  Input   : {original}")
        print(f"     Got    : {got_text}")
        print(f"     Expect : {expected_text}")
        print(f"     Values : {got_vals} (expected {expected_vals})")
        print()
        if not (ok_text and ok_vals):
            all_passed = False

    if all_passed:
        print("All unit tests passed. ✓\n")
    else:
        print("Some tests FAILED. ✗\n")

    # ── Check dataset samples ─────────────────────────────────────────────────
    for i in [0, 1, 5]:
        sample = dataset[i]
        n_nums = sample["number_mask"].sum().item()
        print(f"── Sample {i} ─────────────────────────────────────────")
        print(f"  target          : {sample['target'][:80]}")
        print(f"  n_[NUM]_tokens  : {n_nums}")
        # Show number values where mask is True
        vals = sample["number_values"][sample["number_mask"]].tolist()
        real_vals = [round(denormalize_number(v), 3) for v in vals]
        print(f"  number values   : {real_vals}")
        print(f"  input_ids shape : {sample['input_ids'].shape}")
        print()

    # ── DataLoader test ───────────────────────────────────────────────────────
    print("── DataLoader test ─────────────────────────────────────")
    loader = DataLoader(dataset, batch_size=4, collate_fn=xval_collate_fn, shuffle=False)
    batch  = next(iter(loader))
    print(f"  input_ids shape     : {batch['input_ids'].shape}")
    print(f"  number_values shape : {batch['number_values'].shape}")
    print(f"  number_mask shape   : {batch['number_mask'].shape}")
    print(f"  total [NUM] tokens  : {batch['number_mask'].sum().item()}")
    print("\n✓ xVal Dataset OK")
