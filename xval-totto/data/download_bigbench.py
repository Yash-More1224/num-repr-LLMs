"""
data/download_bigbench.py
==========================
Download and prepare the BigBench Arithmetic dataset.

The BigBench main arithmetic task tests all four basic arithmetic operations
(+, -, *, /) using natural language questions with numerical answers.

Format of raw task.json:
  {
    "examples": [
      {"input": "What is 3 + 4?", "target_scores": {"7": 1, "6": 0, "8": 0}},
      {"input": "What is 12 * 12?", "target_scores": {"144": 1, "143": 0, ...}},
      ...
    ]
  }

We convert this to three JSONL files:
  - train.jsonl  (80%)
  - dev.jsonl    (10%)
  - test.jsonl   (10%)

Each JSONL line:
  {"id": "arith_00001", "input": "What is 3 + 4?", "target": "7"}

Output location: numeric-eval/data/bigbench_arithmetic/
  (parent directory of xval-totto, as requested by user)

Usage
-----
  python data/download_bigbench.py \\
      --output_dir ../data/bigbench_arithmetic \\
      --seed 42
"""

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

BIGBENCH_ARITHMETIC_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/"
    "bigbench/benchmark_tasks/arithmetic/task.json"
)


def download_task(url: str) -> dict:
    """Download and parse the BigBench arithmetic task.json."""
    print(f"Downloading: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        print("Please download manually from:", file=sys.stderr)
        print(f"  {url}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(raw)
    print(f"Downloaded task: '{data.get('name', '?')}'")
    print(f"Description    : {data.get('description', '?')}")
    return data


def extract_examples(data: dict) -> list:
    """
    Extract (input, target) pairs from BigBench task.json.

    Handles both formats:
      - {"input": "...", "target": "7"}              (text-to-text)
      - {"input": "...", "target_scores": {"7": 1}}  (multiple choice)
    """
    examples = data.get("examples", [])
    records  = []
    skipped  = 0

    for ex in examples:
        inp = ex.get("input", "").strip()
        if not inp:
            skipped += 1
            continue

        # Text-to-text format
        if "target" in ex:
            target = str(ex["target"]).strip()
        # Multiple-choice format: pick the key with score == 1
        elif "target_scores" in ex:
            scores = ex["target_scores"]
            correct = [k for k, v in scores.items() if v == 1]
            if not correct:
                skipped += 1
                continue
            target = correct[0].strip()
        else:
            skipped += 1
            continue

        records.append({"input": inp, "target": target})

    print(f"Extracted {len(records):,} examples  ({skipped} skipped)")
    return records


def split_and_save(records: list, output_dir: Path, seed: int):
    """Shuffle and split records into train/dev/test, then save as JSONL."""
    rng = random.Random(seed)
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
                rec_out = {
                    "id":     f"arith_{i:05d}",
                    "input":  rec["input"],
                    "target": rec["target"],
                }
                f.write(json.dumps(rec_out) + "\n")
        print(f"  {split_name:5s}: {len(split_records):5,} examples -> {path}")

    # Also save the raw task.json for reference
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare BigBench Arithmetic dataset."
    )
    parser.add_argument(
        "--output_dir",
        default="../data/bigbench_arithmetic",
        help="Where to save the JSONL splits (default: ../data/bigbench_arithmetic)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("BigBench Arithmetic Dataset Downloader")
    print("=" * 60)

    # Download
    data = download_task(BIGBENCH_ARITHMETIC_URL)

    # Save raw task.json
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "task.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Raw task.json   -> {raw_path}")

    # Extract examples
    records = extract_examples(data)

    # Print a few examples for sanity check
    print("\nSample examples:")
    for rec in records[:5]:
        print(f"  IN : {rec['input']}")
        print(f"  OUT: {rec['target']}")
        print()

    # Split and save
    print("Splitting and saving ...")
    split_and_save(records, output_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print(f"Done. Dataset saved to: {output_dir.resolve()}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train models on ToTTo  (train_bpe.py, train_xval.py, etc.)")
    print("  2. Evaluate on BigBench   (eval/eval_bigbench.py)")


if __name__ == "__main__":
    main()
