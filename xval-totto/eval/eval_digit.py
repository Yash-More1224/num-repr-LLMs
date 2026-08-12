"""
eval/eval_digit.py
==================
Evaluate the Digit-GPT2 model on the ToTTo dev set.

Mirrors eval_bpe.py exactly, with two differences:
  1. Uses GPT2Digit + totto_digit tokenizer
  2. Pre-processes the prompt with split_numbers_to_digits() before encoding,
     so the prompt matches what the model was trained on.

Digit sequences are longer than BPE (each digit is a separate token), so
we use a slightly larger max_prompt_len by default.

Usage:
  python eval/eval_digit.py \\
      --checkpoint checkpoints/digit/digit_best.pt \\
      --config configs/model_config.yaml \\
      --train_config configs/train_config.yaml \\
      --output results/digit_results.json
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from tokenizers import Tokenizer

from model.gpt2 import GPT2Config
from model.gpt2_digit import GPT2Digit
from tokenizer.train_digit_tokenizer import split_numbers_to_digits
from eval.metrics import (extract_numbers, numeric_exact_match, numeric_mae,
                           numeric_mape, hallucination_rate, aggregate_metrics)


def rejoin_digit_tokens(text: str) -> str:
    """
    Reverse the digit-splitting applied at training time.

    The digit tokenizer separates each digit with a space:
      "84"   → "8 4"    (training)
      "8 4"  → "84"    (this function, applied after decoding)

    Only consecutive single-character digit tokens are merged;
    non-digit tokens break the run.
    """
    parts = text.split(" ")
    out   = []
    buf   = []
    for part in parts:
        if len(part) == 1 and part.isdigit():
            buf.append(part)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            if part:
                out.append(part)
    if buf:
        out.append("".join(buf))
    return " ".join(out)


def build_digit_prompt(example: dict, sep_id: int, tokenizer: Tokenizer,
                       max_seq_len: int) -> torch.Tensor:
    """
    Build the prompt tensor for the Digit model.

    Numbers are pre-split into digit characters before tokenization
    (e.g. "from 1874" -> "from 1 8 7 4"), matching the training procedure.

    Returns: (1, T_prompt) tensor
    """
    linearized = example["linearized"]
    digit_text = split_numbers_to_digits(linearized)

    encoding  = tokenizer.encode(digit_text)
    token_ids = encoding.ids

    try:
        sep_pos    = token_ids.index(sep_id)
        prompt_ids = token_ids[:sep_pos + 1]
    except ValueError:
        prompt_ids = token_ids

    # Digit sequences can be 1.5–2× longer than BPE; use a tighter cap
    # max_prompt_len = max_seq_len - 120   # leave room for generation
    if len(prompt_ids) >= max_seq_len:
        return None

    return torch.tensor([prompt_ids], dtype=torch.long)   # (1, T)


@torch.no_grad()
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f) # convert to a dict.

    # Digit tokenizer is stored separately from BPE tokenizer
    tokenizer_path = train_cfg.get("digit_tokenizer",
                                   "tokenizer/totto_digit/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id    = tokenizer.token_to_id("[EOS]")
    sep_id    = tokenizer.token_to_id("[SEP]")
    print(f"Digit tokenizer loaded from: {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"  EOS={eos_id}, SEP={sep_id}")

    # Vocab size must match the digit tokenizer
    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    config.vocab_size = tokenizer.get_vocab_size()

    model = GPT2Digit(config)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Parameters: {model.num_parameters()/1e6:.2f}M")

    data_path = train_cfg["train_data"] if args.split == "train" else train_cfg["dev_data"]
    examples  = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"Evaluating {len(examples)} {args.split} examples ...")

    per_example_results = []
    skipped_count = 0

    for example in tqdm(examples, desc="Digit eval"):
        reference = example["target"]
        ref_nums  = extract_numbers(reference)

        prompt_tensor = build_digit_prompt(
            example, sep_id, tokenizer,
            train_cfg.get("digit_max_seq_len", config.n_positions)
        )
        if prompt_tensor is None:
            skipped_count += 1
            continue

        prompt_ids = prompt_tensor.to(device)
        prompt_len = prompt_ids.size(1)

        available_tokens = config.n_positions - prompt_len
        if available_tokens <= 0:
            skipped_count += 1
            continue

        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=120,
            eos_token_id=eos_id,
            temperature=0.0,
        )

        gen_ids = output_ids[0, prompt_len:].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]

        # Decode, then rejoin space-separated digits into complete numbers.
        # e.g. tokenizer produces "1 8 7 4" → rejoin gives "1874"
        raw_text  = tokenizer.decode(gen_ids)
        generated = rejoin_digit_tokens(raw_text)

        gen_nums  = extract_numbers(generated)
        em        = numeric_exact_match(ref_nums, gen_nums)
        mae       = numeric_mae(ref_nums, gen_nums)
        mape      = numeric_mape(ref_nums, gen_nums)
        halluc    = hallucination_rate(ref_nums, gen_nums)

        per_example_results.append({
            "example_id":  example.get("example_id", ""),
            "reference":   reference,
            "generated":   generated,
            "ref_nums":    ref_nums,
            "gen_nums":    gen_nums,
            "num_em":      em,
            "num_mae":     mae,
            "num_mape":    mape,
            "halluc":      halluc,
            "count_match": len(gen_nums) == len(ref_nums),
            "n_ref_nums":  len(ref_nums),
            "bleu_ref":    reference,
            "bleu_hyp":    generated,
        })

    summary = aggregate_metrics(per_example_results)
    summary["skipped_count"] = skipped_count

    print(f"\n{'='*60}")
    print(f"Digit-GPT2 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Examples evaluated    : {summary['n_examples']}")
    print(f"  Examples skipped      : {skipped_count}")
    print(f"  BLEU                  : {summary['bleu']:.2f}")
    print(f"  Numeric Exact Match   : {summary['num_em']:.4f}  ({summary['num_em']*100:.1f}%)")
    print(f"  Hallucination Rate    : {summary['halluc_rate']:.4f}  ({summary['halluc_rate']*100:.1f}%)")
    print(f"  Count Match Rate      : {summary['count_match']:.4f}  ({summary['count_match']*100:.1f}%)")
    print(f"  Numeric MAE           : {summary['num_mae']:.4f}")
    print(f"  Numeric MAPE          : {summary['num_mape_pct']:.2f}%")
    print(f"{'='*60}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model": "digit", "checkpoint": args.checkpoint,
                   "summary": summary, "examples": per_example_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--config",         required=True,  help="model_config.yaml")
    parser.add_argument("--train_config",   required=True,  help="train_config.yaml")
    parser.add_argument("--output",         default="results/digit_results.json")
    parser.add_argument("--split",          default="dev", choices=["train", "dev"])
    parser.add_argument("--max_examples",   type=int, default=7700)
    parser.add_argument("--max_new_tokens", type=int, default=120,
                        help="Digit sequences are longer; 120 is safer than 80")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
