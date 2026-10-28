"""
data/preprocess.py
==================
Step 1 of the pipeline. Run this first.

Reads the raw ToTTo .jsonl files and converts each example into a flat
linearized string that GPT-2 can process as a sequence.

Linearization format
--------------------
  [TITLE] <page title> [SEC] <section title> [HEADER] col1 | col2 | col3
  [HL] <col header>: <highlighted cell value> [SEP] <target sentence> [EOS]

The model is trained to predict every token AFTER [SEP].
At inference, the model is given everything up to [SEP] and generates the rest.

Outputs
-------
  processed/train.jsonl          -- one JSON per line for training
  processed/dev.jsonl            -- one JSON per line for dev/eval
  processed/tokenizer_corpus.txt -- all linearized strings, one per line
                                    (used to train the BPE tokenizer)

Each output JSON line has:
  {
    "example_id":   <str>,       original ToTTo example ID
    "linearized":   <str>,       full string: input [SEP] target [EOS]
    "target":       <str>,       just the target sentence (for evaluation)
    "sep_char_pos": <int>        character position of [SEP] in linearized
  }

Usage
-----
  python data/preprocess.py \\
      --data_dir totto_data \\
      --output_dir processed
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


# ── Special tokens ────────────────────────────────────────────────────────────
# These must match exactly what the tokenizer uses.
# They are inserted as literal strings into the linearized text, and the
# tokenizer will recognize them as single tokens during BPE training.
SPECIAL_TOKENS = {
    "title":   "[TITLE]",
    "sec":     "[SEC]",
    "header":  "[HEADER]",
    "hl":      "[HL]",
    "sep":     "[SEP]",
    "eos":     "[EOS]",
    "num":     "[NUM]",   # reserved for xVal; not used in linearization itself
    "pad":     "[PAD]",
    "unk":     "[UNK]",
}


def linearize_example(example: dict) -> tuple:
    """
    Convert one ToTTo example dict into a (linearized_string, target) pair.

    The linearized string has the format:
        [TITLE] ... [SEC] ... [HEADER] ... [HL] col: val [SEP] target [EOS]

    Parameters
    ----------
    example : dict
        One parsed line from totto_train_data.jsonl or totto_dev_data.jsonl.

    Returns
    -------
    linearized : str
        The full linearized string including the target at the end.
    target : str
        Just the target sentence (for evaluation use).
    """
    parts = []

    # ── Page title ────────────────────────────────────────────────────────────
    title = (example.get("table_page_title") or "").strip()
    if title:
        parts.append(f"{SPECIAL_TOKENS['title']} {title}")

    # ── Section title ─────────────────────────────────────────────────────────
    section = (example.get("table_section_title") or "").strip()
    if section:
        parts.append(f"{SPECIAL_TOKENS['sec']} {section}")

    # ── Column headers ────────────────────────────────────────────────────────
    # Use the first row of the table as the header row.
    table = example["table"]
    if table and table[0]:
        headers = [cell["value"].strip() for cell in table[0]]
        header_str = " | ".join(h for h in headers if h)
        if header_str:
            parts.append(f"{SPECIAL_TOKENS['header']} {header_str}")

    # ── Highlighted cells ─────────────────────────────────────────────────────
    # Only the highlighted cells are included (not the whole table).
    # Each cell is prefixed with its column header for context.
    seen_cells = set()  # avoid duplicates when a cell is highlighted twice
    for row_idx, col_idx in example["highlighted_cells"]:
        key = (row_idx, col_idx)
        if key in seen_cells:
            continue
        seen_cells.add(key)

        # Bounds check
        if row_idx >= len(table) or col_idx >= len(table[row_idx]):
            continue

        cell_val = table[row_idx][col_idx]["value"].strip()
        if not cell_val:
            continue

        # Get the column header (from row 0)
        col_header = ""
        if table[0] and col_idx < len(table[0]):
            col_header = table[0][col_idx]["value"].strip()

        # Format: [HL] ColHeader: cell_value
        # If cell_val == col_header (rare), just show the value
        if col_header and col_header != cell_val:
            parts.append(f"{SPECIAL_TOKENS['hl']} {col_header}: {cell_val}")
        else:
            parts.append(f"{SPECIAL_TOKENS['hl']} {cell_val}")

    # ── Target sentence ───────────────────────────────────────────────────────
    # Use the final cleaned sentence from annotations.
    target = example["sentence_annotations"][0]["final_sentence"].strip()

    # ── Assemble full string ──────────────────────────────────────────────────
    input_part = " ".join(parts)
    linearized = (
        f"{input_part} "
        f"{SPECIAL_TOKENS['sep']} "
        f"{target} "
        f"{SPECIAL_TOKENS['eos']}"
    )

    return linearized, target


def process_split(
    input_path,
    output_path,
    corpus_file,
    split_name,
):
    """
    Process one split (train or dev) and write to output_path.

    Parameters
    ----------
    input_path   : path to the raw .jsonl file
    output_path  : path to write the processed .jsonl file
    corpus_file  : open file handle; linearized strings are written here too
                   (used for tokenizer training — train split only, else None)
    split_name   : "train" or "dev" (for logging)

    Returns
    -------
    stats : dict with keys: total, skipped, avg_input_len, avg_target_len
    """
    total = 0
    skipped = 0
    input_lengths = []
    target_lengths = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f, desc=f"Processing {split_name}"):
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Skip examples with no sentence annotations
                if not example.get("sentence_annotations"):
                    skipped += 1
                    continue

                linearized, target = linearize_example(example)

                # Find character position of [SEP] — needed by dataset classes
                # to know where the input ends and the target begins.
                sep_pos = linearized.index(SPECIAL_TOKENS["sep"])

                record = {
                    "example_id":   str(example.get("example_id", total)),
                    "linearized":   linearized,
                    "target":       target,
                    "sep_char_pos": sep_pos,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Write to tokenizer corpus (train split only)
                if corpus_file is not None:
                    corpus_file.write(linearized + "\n")

                # Collect stats
                input_lengths.append(len(linearized[:sep_pos].split()))
                target_lengths.append(len(target.split()))
                total += 1

    avg_in  = sum(input_lengths) / len(input_lengths) if input_lengths else 0
    avg_tgt = sum(target_lengths) / len(target_lengths) if target_lengths else 0

    return {
        "total":          total,
        "skipped":        skipped,
        "avg_input_len":  round(avg_in, 1),
        "avg_target_len": round(avg_tgt, 1),
    }


def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "tokenizer_corpus.txt"

    print("=" * 60)
    print("ToTTo Preprocessor")
    print("=" * 60)
    print(f"Input directory : {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # ── Train split ───────────────────────────────────────────────────────────
    print("Processing TRAIN split ...")
    with open(corpus_path, "w", encoding="utf-8") as corpus_f:
        train_stats = process_split(
            input_path  = data_dir / "totto_train_data.jsonl",
            output_path = output_dir / "train.jsonl",
            corpus_file = corpus_f,
            split_name  = "train",
        )

    # ── Dev split ─────────────────────────────────────────────────────────────
    print("\nProcessing DEV split ...")
    dev_stats = process_split(
        input_path  = data_dir / "totto_dev_data.jsonl",
        output_path = output_dir / "dev.jsonl",
        corpus_file = None,    # don't add dev text to tokenizer corpus
        split_name  = "dev",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Done. Summary:")
    print(f"  Train examples : {train_stats['total']:,}  "
          f"(skipped: {train_stats['skipped']})")
    print(f"  Dev examples   : {dev_stats['total']:,}  "
          f"(skipped: {dev_stats['skipped']})")
    print(f"  Avg input len  : {train_stats['avg_input_len']} words (train)")
    print(f"  Avg target len : {train_stats['avg_target_len']} words (train)")
    print(f"\n  Tokenizer corpus : {corpus_path}")
    print(f"  Corpus lines     : {train_stats['total']:,}")
    print("=" * 60)
    print()
    print("Next step: run  tokenizer/train_tokenizer.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linearize ToTTo dataset for GPT-2 training."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="totto_data",
        help="Directory containing totto_train_data.jsonl and totto_dev_data.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Directory to write processed .jsonl files and tokenizer corpus",
    )
    args = parser.parse_args()
    main(args)
