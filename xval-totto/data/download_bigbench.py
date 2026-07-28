"""
data/download_bigbench.py
==========================
Generate the BigBench Arithmetic dataset locally.

NOTE: The BigBench arithmetic task is PROGRAMMATICALLY GENERATED — its task.json
has "examples": [] because examples are produced by running the task's Python code.
This script generates equivalent arithmetic examples following the same format and
number ranges as the original BigBench task.

Arithmetic operations covered:
  - Addition       : "What is X plus Y?"      → X + Y
  - Subtraction    : "What is X minus Y?"     → X - Y   (result ≥ 0)
  - Multiplication : "What is X times Y?"     → X * Y
  - Division       : "What is X divided by Y?" → X / Y  (integer, divisible pairs)

Each JSONL line:
  {"id": "arith_00001", "input": "What is 3 plus 4?", "target": "7"}

Output: ../data/bigbench_arithmetic/  (parent directory of xval-totto)

Usage
-----
  cd ~/numeric-eval/xval-totto
  python data/download_bigbench.py --output_dir ../data/bigbench_arithmetic --n_examples 5000
"""

import argparse
import json
import random
from pathlib import Path


# ── Arithmetic generators ─────────────────────────────────────────────────────

def gen_addition(rng: random.Random) -> dict:
    a = rng.randint(0, 1000)
    b = rng.randint(0, 1000)
    return {"input": f"What is {a} plus {b}?", "target": str(a + b)}


def gen_subtraction(rng: random.Random) -> dict:
    a = rng.randint(0, 1000)
    b = rng.randint(0, 1000)
    a, b = max(a, b), min(a, b)   # ensure non-negative result
    return {"input": f"What is {a} minus {b}?", "target": str(a - b)}


def gen_multiplication(rng: random.Random) -> dict:
    a = rng.randint(0, 100)
    b = rng.randint(0, 100)
    return {"input": f"What is {a} times {b}?", "target": str(a * b)}


def gen_division(rng: random.Random) -> dict:
    """Integer division only — b ≠ 0, a always divisible by b."""
    b = rng.randint(1, 100)
    quotient = rng.randint(0, 100)
    a = b * quotient
    return {"input": f"What is {a} divided by {b}?", "target": str(quotient)}


GENERATORS = [gen_addition, gen_subtraction, gen_multiplication, gen_division]


# ── Generate examples ─────────────────────────────────────────────────────────

def generate_examples(n: int, seed: int) -> list:
    """
    Generate n unique arithmetic examples, balanced across 4 operations.
    De-duplicates on the input question to avoid repeated prompts.
    """
    rng     = random.Random(seed)
    seen    = set()
    records = []

    max_attempts = n * 20
    attempts     = 0

    while len(records) < n and attempts < max_attempts:
        attempts += 1
        gen_fn = rng.choice(GENERATORS)
        rec    = gen_fn(rng)
        key    = rec["input"]
        if key not in seen:
            seen.add(key)
            records.append(rec)

    if len(records) < n:
        print(f"[WARN] Only generated {len(records):,} unique examples "
              f"(requested {n}). Increase max_val ranges if you need more.")

    print(f"Generated {len(records):,} unique arithmetic examples  (seed={seed})")
    return records


# ── Split and save ────────────────────────────────────────────────────────────

def split_and_save(records: list, output_dir: Path, seed: int):
    """Shuffle and split 80/10/10, then write JSONL files."""
    rng = random.Random(seed + 1)
    rng.shuffle(records)

    n       = len(records)
    n_train = int(0.8 * n)
    n_dev   = int(0.1 * n)

    splits = {
        "train": records[:n_train],
        "dev":   records[n_train:n_train + n_dev],
        "test":  records[n_train + n_dev:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(split_records):
                out = {
                    "id":     f"arith_{i:05d}",
                    "input":  rec["input"],
                    "target": rec["target"],
                }
                f.write(json.dumps(out) + "\n")
        print(f"  {split_name:5s}: {len(split_records):5,} examples → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate BigBench-style Arithmetic dataset (locally)."
    )
    parser.add_argument(
        "--output_dir",
        default="../data/bigbench_arithmetic",
        help="Where to save the JSONL splits (default: ../data/bigbench_arithmetic)",
    )
    parser.add_argument(
        "--n_examples", type=int, default=5000,
        help="Total number of arithmetic examples to generate (default: 5000). "
             "Split 80/10/10 into train/dev/test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("BigBench Arithmetic Dataset Generator")
    print("=" * 60)
    print(f"  Generating   : {args.n_examples:,} examples")
    print(f"  Operations   : addition, subtraction, multiplication, division")
    print(f"  Output dir   : {output_dir.resolve()}")
    print(f"  Seed         : {args.seed}")
    print()

    records = generate_examples(args.n_examples, seed=args.seed)

    print("\nSample examples:")
    for rec in records[:4]:
        print(f"  IN : {rec['input']}")
        print(f"  OUT: {rec['target']}")
    print()

    print("Splitting and saving ...")
    split_and_save(records, output_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print(f"Done. Dataset saved to: {output_dir.resolve()}")
    print("=" * 60)
    print("\nNext steps:")
    print("  PYTHONPATH=. torchrun --nproc_per_node=4 train/train_bigbench.py --model_type bpe ...")
    print("  python eval/eval_bigbench.py --model_type bpe ...")


if __name__ == "__main__":
    main()
