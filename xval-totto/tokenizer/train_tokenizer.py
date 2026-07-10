"""
tokenizer/train_tokenizer.py
============================
Step 2 of the pipeline. Run this after preprocess.py.

Trains a Byte-Pair Encoding (BPE) tokenizer on the ToTTo linearized text
and saves it to disk. Both models (BPE-GPT2 and xVal-GPT2) will use this
SAME tokenizer — the only difference is that xVal replaces numbers with
the [NUM] special token before tokenizing.

Why train our own tokenizer instead of using GPT-2's?
------------------------------------------------------
1. We want full control over the vocabulary so both models are guaranteed
   identical tokenization for all non-numeric text.
2. GPT-2's tokenizer has 50,257 tokens — overkill for ToTTo's domain.
   A 16K vocab is smaller, trains faster, and is sufficient here.
3. We need [NUM] to be a proper special token (single token, never split).

Output
------
  tokenizer/totto_bpe/     -- saved tokenizer directory (HuggingFace format)
    tokenizer.json
    special_tokens_map.json
    tokenizer_config.json

Usage
-----
  python tokenizer/train_tokenizer.py \\
      --corpus processed/tokenizer_corpus.txt \\
      --output_dir tokenizer/totto_bpe \\
      --vocab_size 16000

Loading later in training scripts
----------------------------------
  from tokenizers import Tokenizer
  tokenizer = Tokenizer.from_file("tokenizer/totto_bpe/tokenizer.json")
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


# ── Special tokens ────────────────────────────────────────────────────────────
# Order matters here: IDs will be 0, 1, 2, ... in this order.
# Keep this list identical across both model training scripts.
SPECIAL_TOKENS = [
    "[PAD]",     # 0  — padding token
    "[UNK]",     # 1  — unknown token
    "[EOS]",     # 2  — end of sequence
    "[SEP]",     # 3  — separates input from target
    "[TITLE]",   # 4  — page title prefix
    "[SEC]",     # 5  — section title prefix
    "[HEADER]",  # 6  — column headers prefix
    "[HL]",      # 7  — highlighted cell prefix
    "[NUM]",     # 8  — numeric placeholder (used by xVal model only)
]

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def train_tokenizer(corpus_path: Path, output_dir: Path, vocab_size: int):
    """
    Train a BPE tokenizer on the corpus and save it.

    Parameters
    ----------
    corpus_path : Path to tokenizer_corpus.txt (one sentence per line)
    output_dir  : Directory to save the trained tokenizer
    vocab_size  : Target vocabulary size (default 16000)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus  : {corpus_path}")
    print(f"Vocab   : {vocab_size:,} tokens")
    print(f"Output  : {output_dir}")
    print()

    # ── 1. Initialize BPE model ───────────────────────────────────────────────
    # unk_token is required by BPE — used for characters unseen during training.
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # ── 2. Pre-tokenizer ──────────────────────────────────────────────────────
    # Whitespace splits on whitespace boundaries before BPE merges are applied.
    # This means BPE merges never cross whitespace, keeping word-level structure.
    # Our special tokens (e.g. [TITLE]) are single space-separated words, so
    # they will be handled correctly by the trainer's special_tokens list.
    tokenizer.pre_tokenizer = Whitespace()

    # ── 3. Trainer ────────────────────────────────────────────────────────────
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,  # always get single-token IDs
        min_frequency=2,                # ignore pairs seen fewer than 2 times
        show_progress=True,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("Training BPE tokenizer ...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    print(f"Actual vocab size: {tokenizer.get_vocab_size():,}")

    # ── 5. Verify special token IDs ───────────────────────────────────────────
    vocab = tokenizer.get_vocab()
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:12s} -> {vocab.get(tok, 'NOT FOUND')}")

    # Sanity checks
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert vocab.get(tok) == i, (
            f"Special token {tok!r} expected ID {i}, got {vocab.get(tok)}. "
            "This would break the model — check trainer configuration."
        )
    print("\nAll special token IDs verified. ✓")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved to: {tokenizer_path}")

    # Also save a simple config file that our training scripts will read.
    config = {
        "vocab_size":     tokenizer.get_vocab_size(),
        "pad_token":      PAD_TOKEN,
        "pad_token_id":   vocab[PAD_TOKEN],
        "unk_token":      UNK_TOKEN,
        "unk_token_id":   vocab[UNK_TOKEN],
        "eos_token":      "[EOS]",
        "eos_token_id":   vocab["[EOS]"],
        "sep_token":      "[SEP]",
        "sep_token_id":   vocab["[SEP]"],
        "num_token":      "[NUM]",
        "num_token_id":   vocab["[NUM]"],
        "special_tokens": SPECIAL_TOKENS,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to  : {config_path}")

    # ── 7. Quick smoke test ───────────────────────────────────────────────────
    print("\nSmoke test:")
    test_strings = [
        "[TITLE] List of wildfires [SEC] 2018 [HEADER] Name | Acres [HL] Acres: 281,893 [SEP] Thomas burned 281,893 acres. [EOS]",
        "[TITLE] 78th Congress [HL] Date: November 22, 1943 [SEP] Barbour served until November 22, 1943. [EOS]",
        "[NUM] is the xVal placeholder token.",
    ]
    for s in test_strings:
        enc = tokenizer.encode(s)
        decoded = tokenizer.decode(enc.ids)
        print(f"  Input   : {s[:70]}...")
        print(f"  Tokens  : {enc.tokens[:12]} ...")
        print(f"  n_tokens: {len(enc.ids)}")
        print()

    return tokenizer


def main(args):
    corpus_path = Path(args.corpus)
    output_dir  = Path(args.output_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}\n"
            "Run data/preprocess.py first."
        )

    print("=" * 60)
    print("BPE Tokenizer Trainer")
    print("=" * 60)

    train_tokenizer(
        corpus_path=corpus_path,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
    )

    print("=" * 60)
    print("Done.")
    print("=" * 60)
    print()
    print("Next step: run  data/dataset.py  (to verify data loading)")
    print("           then start model/gpt2_bpe.py  and  model/gpt2_xval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on ToTTo linearized text."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="processed/tokenizer_corpus.txt",
        help="Path to tokenizer_corpus.txt produced by preprocess.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer/totto_bpe",
        help="Directory to save the trained tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16000,
        help="Target BPE vocabulary size (default: 16000)",
    )
    args = parser.parse_args()
    main(args)
