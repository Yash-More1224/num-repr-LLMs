"""
eval/eval_bpe.py
================
Evaluate the BPE-GPT2 model on the ToTTo dev set.

For each dev example:
  1. Encode the INPUT portion (before [SEP]) as a prompt
  2. Generate up to 80 tokens using greedy decoding
  3. Decode the generated tokens to text
  4. Extract numbers from the generated text using regex
  5. Compare against reference numbers with all numeric metrics

Saves results to: results/totto/bpe.json

Usage:
  python eval/eval_bpe.py \\
      --checkpoint checkpoints/bpe/bpe_epoch05_step004715.pt \\
      --config configs/model_config.yaml \\
      --train_config configs/train_config.yaml \\
      --output results/totto/bpe.json \\
      --max_examples 7700       # use full dev set; set lower for quick test
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from tokenizers import Tokenizer

from model.gpt2 import GPT2Config
from model.gpt2_bpe import GPT2BPE
from eval.metrics import extract_numbers, numeric_exact_match, numeric_mae, \
    numeric_mape, hallucination_rate, aggregate_metrics


def build_prompt(example: dict, sep_token_id: int, tokenizer: Tokenizer,
                 max_seq_len: int) -> torch.Tensor:
    """
    From a processed dev example, build the prompt tensor (input up to [SEP]).

    The prompt is: everything in 'linearized' up to and including [SEP].
    We feed this into the model and let it generate the target.

    Returns: (1, T_prompt) tensor of token IDs
    """
    linearized = example["linearized"]

    # Tokenize the full string first to ensure identical tokenization to training
    encoding = tokenizer.encode(linearized)
    token_ids = encoding.ids

    # Find the position of [SEP]
    try:
        sep_pos = token_ids.index(sep_token_id)
        prompt_ids = token_ids[:sep_pos + 1]
    except ValueError:
        prompt_ids = token_ids

    # Truncate from the left if the prompt is too long, so we don't chop off [SEP]
    max_prompt_len = max_seq_len - 80
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    return torch.tensor([prompt_ids], dtype=torch.long)   # (1, T)


def decode_generated(output_ids: torch.Tensor, prompt_len: int,
                     tokenizer: Tokenizer, eos_id: int) -> str:
    """
    Decode the generated portion of output_ids (after the prompt).

    Stops at [EOS] token if present.

    Returns the generated text as a plain string.
    """
    gen_ids = output_ids[0, prompt_len:].tolist()

    # Truncate at EOS
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    return tokenizer.decode(gen_ids)


@torch.no_grad()
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load configs ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])
    if tokenizer.decoder is None:
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        tokenizer.decoder = MetaspaceDecoder(replacement=" ")
    eos_id    = tokenizer.token_to_id("[EOS]")
    sep_id    = tokenizer.token_to_id("[SEP]")
    print(f"Tokenizer loaded. EOS={eos_id}, SEP={sep_id}")

    # ── Load model ────────────────────────────────────────────────────────────
    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    model = GPT2BPE(config)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # ── Load data (train or dev split) ──────────────────────────────────────
    data_path = train_cfg["train_data"] if args.split == "train" else train_cfg["dev_data"]
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"Evaluating {len(examples)} {args.split} examples ...")

    # ── Generation loop ───────────────────────────────────────────────────────
    per_example_results = []

    for example in tqdm(examples, desc="BPE eval"):
        reference   = example["target"]
        ref_nums    = extract_numbers(reference)

        # Build prompt
        prompt_ids  = build_prompt(example, sep_id, tokenizer,
                                   train_cfg["max_seq_len"]).to(device)
        prompt_len  = prompt_ids.size(1)

        # Generate (greedy, no repetition penalty)
        output_ids  = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            temperature=0.0,
            repetition_penalty=1.0,
        )

        # Decode generated portion
        generated   = decode_generated(output_ids, prompt_len, tokenizer, eos_id)
        gen_nums    = extract_numbers(generated)

        # Compute per-example metrics
        em    = numeric_exact_match(ref_nums, gen_nums)
        mae   = numeric_mae(ref_nums, gen_nums)
        mape  = numeric_mape(ref_nums, gen_nums)
        halluc = hallucination_rate(ref_nums, gen_nums)
        count_match = len(gen_nums) == len(ref_nums)

        per_example_results.append({
            "example_id":   example.get("example_id", ""),
            "reference":    reference,
            "generated":    generated,
            "ref_nums":     ref_nums,
            "gen_nums":     gen_nums,
            "num_em":       em,
            "num_mae":      mae,
            "num_mape":     mape,
            "halluc":       halluc,
            "count_match":  count_match,
            "n_ref_nums":   len(ref_nums),
            "bleu_ref":     reference,
            "bleu_hyp":     generated,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_metrics(per_example_results)

    print(f"\n{'='*60}")
    print(f"BPE-GPT2 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Examples evaluated    : {summary['n_examples']}")
    print(f"  Examples with numbers : {summary['n_with_numbers']}")
    print(f"  BLEU                  : {summary['bleu']:.2f}")
    print(f"  Numeric Exact Match   : {summary['num_em']:.4f}  ({summary['num_em']*100:.1f}%)")
    print(f"  Hallucination Rate    : {summary['halluc_rate']:.4f}  ({summary['halluc_rate']*100:.1f}%)")
    print(f"  Count Match Rate      : {summary['count_match']:.4f}  ({summary['count_match']*100:.1f}%)")
    print(f"  Numeric MAE           : {summary['num_mae']:.4f}")
    print(f"  Numeric MAPE          : {summary['num_mape_pct']:.2f}%")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model":    "bpe",
            "checkpoint": args.checkpoint,
            "summary":  summary,
            "examples": per_example_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--config",        required=True,  help="model_config.yaml")
    parser.add_argument("--train_config",  required=True,  help="train_config.yaml")
    parser.add_argument("--output",        default="results/totto/bpe.json")
    parser.add_argument("--split",         default="dev", choices=["train", "dev"],
                        help="Which split to evaluate on: 'train' or 'dev' (default: dev)")
    parser.add_argument("--max_examples",  type=int, default=7700,
                        help="Number of examples to evaluate (7700 = full dev set)")
    parser.add_argument("--max_new_tokens",type=int, default=80)
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
