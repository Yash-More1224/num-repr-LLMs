"""
tokenizer/train_digit_tokenizer.py
===================================
Train a BPE tokenizer where digits 0-9 are guaranteed single tokens.

The key idea of digit tokenization:
  - Regular BPE may merge "18" into a single token, losing individual digit access.
  - Here, we pre-tokenize numbers into space-separated digits BEFORE training,
    so BPE never merges individual digit characters.
  - e.g., "1874" becomes "1 8 7 4" in the training corpus → each digit
    is a separate token, never merged by BPE.

The digit tokenizer shares ALL special tokens with the BPE tokenizer
([PAD], [UNK], [EOS], [SEP], [TITLE], [SEC], [HEADER], [HL], [NUM])
for compatibility with the same preprocessing pipeline.

Usage
-----
  python tokenizer/train_digit_tokenizer.py \\
      --corpus processed/tokenizer_corpus.txt \\
      --output_dir tokenizer/totto_digit \\
      --vocab_size 16000

Note: Run preprocess.py first to generate the corpus file.
The digit corpus is auto-generated from the BPE corpus by splitting numbers.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Reuse the same special tokens (and their IDs 0-8) as the BPE tokenizer.
# This is REQUIRED so the dataset preprocessing scripts work unchanged.
SPECIAL_TOKENS = [
    "[PAD]",     # 0
    "[UNK]",     # 1
    "[EOS]",     # 2
    "[SEP]",     # 3
    "[TITLE]",   # 4
    "[SEC]",     # 5
    "[HEADER]",  # 6
    "[HL]",      # 7
    "[NUM]",     # 8
]

# Regex to find number spans (same as dataset_xval.py)
NUMBER_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b")


def split_numbers_to_digits(text: str) -> str:
    """
    Replace every number span in text with its space-separated digit form.

    Examples:
        "1874"    -> "1 8 7 4"
        "281,893" -> "2 8 1 8 9 3"   (commas removed first)
        "3.14"    -> "3 . 1 4"       (decimal point kept as separate char)
        "the 42nd" -> "the 4 2nd"    (only numeric chars separated)
    """
    def replace_num(match):
        raw = match.group(1)
        # Remove commas for comma-grouped numbers
        raw = raw.replace(",", "")
        # Split each character with spaces (keeps the '.' for decimals)
        return " ".join(raw)

    return NUMBER_RE.sub(replace_num, text)


def build_digit_corpus(bpe_corpus_path: Path, digit_corpus_path: Path):
    """
    Create a digit-tokenized corpus from the BPE corpus by splitting all
    number spans into individual digit characters.
    """
    print(f"Building digit corpus from: {bpe_corpus_path}")
    n_lines = 0
    with open(bpe_corpus_path, "r", encoding="utf-8") as src, \
         open(digit_corpus_path, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(split_numbers_to_digits(line))
            n_lines += 1
    print(f"  Wrote {n_lines:,} lines to: {digit_corpus_path}")


def train_digit_tokenizer(corpus_path: Path, output_dir: Path, vocab_size: int):
    """
    Train and save the digit-aware BPE tokenizer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build digit corpus in a temp file
    digit_corpus_path = output_dir / "digit_corpus.txt"
    build_digit_corpus(corpus_path, digit_corpus_path)

    print(f"\nTraining digit BPE tokenizer ...")
    print(f"  Corpus  : {digit_corpus_path}")
    print(f"  Vocab   : {vocab_size:,}")
    print(f"  Output  : {output_dir}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size     = vocab_size,
        special_tokens = SPECIAL_TOKENS,
        min_frequency  = 2,
        show_progress  = True,
    )

    tokenizer.train(files=[str(digit_corpus_path)], trainer=trainer)
    print(f"Actual vocab size: {tokenizer.get_vocab_size():,}")

    # Verify special token IDs
    vocab = tokenizer.get_vocab()
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:12s} -> {vocab.get(tok, 'NOT FOUND')}")
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert vocab.get(tok) == i, (
            f"Special token {tok!r} expected ID {i}, got {vocab.get(tok)}."
        )
    print("All special token IDs verified. ✓")

    # Verify digit characters are single tokens (the key invariant)
    for d in "0123456789":
        assert d in vocab, f"Digit '{d}' not in vocabulary — digit tokenization broken!"
    print("Digits 0-9 are all single tokens. ✓")

    # Verify decimal point is also a token
    if "." in vocab:
        print(f"Decimal point '.' is a single token (ID {vocab['.']}). ✓")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved to: {tokenizer_path}")

    # Save config
    config = {
        "vocab_size":     tokenizer.get_vocab_size(),
        "pad_token":      "[PAD]",
        "pad_token_id":   vocab["[PAD]"],
        "unk_token":      "[UNK]",
        "unk_token_id":   vocab["[UNK]"],
        "eos_token":      "[EOS]",
        "eos_token_id":   vocab["[EOS]"],
        "sep_token":      "[SEP]",
        "sep_token_id":   vocab["[SEP]"],
        "num_token":      "[NUM]",
        "num_token_id":   vocab["[NUM]"],
        "special_tokens": SPECIAL_TOKENS,
        "digit_tokenizer": True,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to  : {config_path}")

    # Smoke test
    print("\nSmoke test:")
    test_cases = [
        ("1874", "1 8 7 4"),
        ("76th Governor", None),
        ("[SEP] the 76th Governor [EOS]", None),
    ]
    for text, _ in test_cases:
        digit_text = split_numbers_to_digits(text)
        enc = tokenizer.encode(digit_text)
        print(f"  '{text}' -> '{digit_text}' -> tokens: {enc.tokens[:10]}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a digit-aware BPE tokenizer for digit tokenization approach."
    )
    parser.add_argument("--corpus",     default="processed/tokenizer_corpus.txt")
    parser.add_argument("--output_dir", default="tokenizer/totto_digit")
    parser.add_argument("--vocab_size", type=int, default=16000)
    args = parser.parse_args()

    train_digit_tokenizer(
        corpus_path = Path(args.corpus),
        output_dir  = Path(args.output_dir),
        vocab_size  = args.vocab_size,
    )
    print("\n✓ Digit tokenizer training complete.")
    print("Next: run train/train_digit.py with --config pointing to digit tokenizer.")


if __name__ == "__main__":
    main()
