"""
eval/eval_xval.py
=================
Evaluate the xVal-GPT2 model on the ToTTo dev set.

For each dev example:
  1. Encode the INPUT portion (before [SEP]) as a prompt
     - Also build number_values and number_mask tensors for the prompt
  2. Generate up to 80 tokens using greedy decoding
     - model.generate() returns generated token IDs AND predicted scalar values
       for each [NUM] token in the generated portion
  3. Denormalize scalar predictions to get actual number values
  4. Reconstruct the generated sentence by substituting predicted values
     for [NUM] tokens
  5. Compute all the same metrics as BPE, PLUS:
     - Scalar MSE (log-normalized): how accurate is the regression head?

Saves results to: results/xval_results.json

Usage:
  python eval/eval_xval.py \\
      --checkpoint checkpoints/xval/xval_epoch04_step003772.pt \\
      --config configs/model_config.yaml \\
      --train_config configs/train_config.yaml \\
      --output results/xval_results.json \\
      --max_examples 7700
"""

import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder

from model.gpt2 import GPT2Config
from model.gpt2_xval import GPT2XVal
from data.dataset_xval import replace_numbers_with_num_token, denormalize_number
from eval.metrics import extract_numbers, numeric_exact_match, numeric_mae, \
    numeric_mape, hallucination_rate, aggregate_metrics


# ── Number formatting ─────────────────────────────────────────────────────────

def format_number(value: float) -> str:
    """
    Format a predicted float back to a human-readable string.

    Strategy:
      - If value is close to an integer, format as integer (e.g., 2018 not 2018.0)
      - Otherwise, use up to 4 significant digits
    """
    if abs(value - round(value)) < 0.5:
        return str(int(round(value)))
    else:
        return f"{value:.4g}"


# ── Prompt builder (xVal version) ─────────────────────────────────────────────

def build_xval_prompt(
    example:      dict,
    tokenizer:    Tokenizer,
    num_token_id: int,
    max_seq_len:  int,
    sep_token_id: int,
) -> tuple:
    """
    Build prompt tensors for the xVal model.

    Returns:
      prompt_ids     : (1, T_prompt) int64 — token IDs with [NUM] for numbers
      number_values  : (1, T_prompt) float32 — log-normalized values at [NUM] positions
      number_mask    : (1, T_prompt) bool — True at [NUM] positions
    """
    linearized = example["linearized"]

    # Replace numbers with [NUM] and get their values on the FULL string
    xval_text, num_values_ordered = replace_numbers_with_num_token(linearized)

    # Tokenize the full string first to ensure identical tokenization to training
    encoding = tokenizer.encode(xval_text)
    token_ids = encoding.ids

    # Find the position of [SEP]
    try:
        sep_pos = token_ids.index(sep_token_id)
        prompt_ids = token_ids[:sep_pos + 1]
    except ValueError:
        prompt_ids = token_ids

    # Build full number_values and number_mask tensors for the prompt
    T_prompt = len(prompt_ids)
    nv_prompt = torch.zeros(1, T_prompt)
    nm_prompt = torch.zeros(1, T_prompt, dtype=torch.bool)

    num_idx = 0
    for t, tok_id in enumerate(prompt_ids):
        if tok_id == num_token_id and num_idx < len(num_values_ordered):
            nv_prompt[0, t] = num_values_ordered[num_idx]
            nm_prompt[0, t] = True
            num_idx += 1

    # Truncate from the left if the prompt is too long, to keep [SEP]
    max_prompt_len = max_seq_len - 80
    if T_prompt > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
        nv = nv_prompt[:, -max_prompt_len:]
        nm = nm_prompt[:, -max_prompt_len:]
    else:
        nv = nv_prompt
        nm = nm_prompt

    return torch.tensor([prompt_ids], dtype=torch.long), nv, nm


def reconstruct_text(gen_ids: list, pred_values: list,
                     tokenizer: Tokenizer, num_token_id: int,
                     eos_id: int) -> tuple:
    """
    Decode the generated token sequence and substitute [NUM] tokens
    with the predicted scalar values.

    Parameters
    ----------
    gen_ids     : list of token IDs (generated portion only, after prompt)
    pred_values : list of log-normalized predicted values for [NUM] tokens
    tokenizer   : BPE tokenizer
    num_token_id: token ID for [NUM]
    eos_id      : stop at this token

    Returns
    -------
    reconstructed_text : str — the full sentence with actual numbers
    gen_numbers        : list of float — the predicted numbers in order
    raw_pred_values    : list of float — log-normalized values (for MSE computation)
    """
    # Truncate at EOS
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    # Walk through tokens to collect predicted numbers
    num_idx = 0
    gen_numbers = []
    raw_pred_values = []
    
    # Decode the full string to preserve BPE spacing
    reconstructed_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    for tok_id in gen_ids:
        if tok_id == num_token_id and num_idx < len(pred_values):
            log_val = pred_values[num_idx]
            actual_val = denormalize_number(log_val)
            gen_numbers.append(actual_val)
            raw_pred_values.append(log_val)
            
            # Splice predicted scalar values back into [NUM] placeholders
            reconstructed_text = reconstructed_text.replace("[NUM]", format_number(actual_val), 1)
            num_idx += 1

    reconstructed_text = reconstructed_text.strip()
    return reconstructed_text, gen_numbers, raw_pred_values


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
    tokenizer    = Tokenizer.from_file(train_cfg["tokenizer"])
    if tokenizer.decoder is None:
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        tokenizer.decoder = MetaspaceDecoder(replacement=" ")
    eos_id       = tokenizer.token_to_id("[EOS]")
    sep_id       = tokenizer.token_to_id("[SEP]")
    num_token_id = model_cfg.get("num_token_id", 8)
    print(f"Tokenizer loaded. EOS={eos_id}, SEP={sep_id}, [NUM]={num_token_id}")

    # ── Load model ────────────────────────────────────────────────────────────
    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    model = GPT2XVal(
        config       = config,
        num_token_id = num_token_id,
        mse_lambda   = model_cfg.get("mse_lambda", 1.0),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # ── Load dev data ─────────────────────────────────────────────────────────
    dev_examples = []
    with open(train_cfg["dev_data"]) as f:
        for line in f:
            dev_examples.append(json.loads(line))

    if args.max_examples > 0:
        dev_examples = dev_examples[:args.max_examples]
    print(f"Evaluating {len(dev_examples)} dev examples ...")

    # ── Generation loop ───────────────────────────────────────────────────────
    per_example_results = []

    # For scalar MSE computation
    all_pred_log_vals = []
    all_true_log_vals = []

    for example in tqdm(dev_examples, desc="xVal eval"):
        reference = example["target"]
        ref_nums  = extract_numbers(reference)

        # Get true log-normalized values for reference numbers
        # (for scalar MSE — compare what the model predicted vs what it should have)
        _, true_log_vals = replace_numbers_with_num_token(reference)

        # Build prompt tensors
        prompt_ids, nv, nm = build_xval_prompt(
            example, tokenizer, num_token_id, train_cfg["max_seq_len"], sep_id
        )
        prompt_ids = prompt_ids.to(device)
        nv         = nv.to(device)
        nm         = nm.to(device)
        prompt_len = prompt_ids.size(1)

        # Generate (greedy, no repetition penalty)
        gen_ids, pred_values = model.generate(
            prompt_ids, nv, nm,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            temperature=0.0,
            repetition_penalty=1.0,
        )
        pred_values = pred_values[0]   # list of log-normalized floats

        # Reconstruct generated sentence
        gen_ids = gen_ids[0, prompt_len:].tolist()
        generated, gen_nums, raw_pred_vals = reconstruct_text(
            gen_ids, pred_values, tokenizer, num_token_id, eos_id
        )

        # Collect for scalar MSE
        n_pairs = min(len(true_log_vals), len(raw_pred_vals))
        all_true_log_vals.extend(true_log_vals[:n_pairs])
        all_pred_log_vals.extend(raw_pred_vals[:n_pairs])

        # Compute per-example metrics
        em         = numeric_exact_match(ref_nums, gen_nums)
        mae        = numeric_mae(ref_nums, gen_nums)
        mape       = numeric_mape(ref_nums, gen_nums)
        halluc     = hallucination_rate(ref_nums, gen_nums)
        count_match = len(gen_nums) == len(ref_nums)

        per_example_results.append({
            "example_id":       example.get("example_id", ""),
            "reference":        reference,
            "generated":        generated,
            "ref_nums":         ref_nums,
            "gen_nums":         gen_nums,
            "pred_log_vals":    raw_pred_vals,
            "true_log_vals":    true_log_vals,
            "num_em":           em,
            "num_mae":          mae,
            "num_mape":         mape,
            "halluc":           halluc,
            "count_match":      count_match,
            "n_ref_nums":       len(ref_nums),
            "bleu_ref":         reference,
            "bleu_hyp":         generated,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_metrics(per_example_results)

    # Scalar MSE (xVal-specific)
    if all_pred_log_vals:
        scalar_mse = sum((p - t) ** 2 for p, t in
                         zip(all_pred_log_vals, all_true_log_vals)) / len(all_pred_log_vals)
        summary["scalar_mse_lognorm"] = round(scalar_mse, 6)
    else:
        summary["scalar_mse_lognorm"] = float("nan")

    print(f"\n{'='*60}")
    print(f"xVal-GPT2 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Examples evaluated    : {summary['n_examples']}")
    print(f"  Examples with numbers : {summary['n_with_numbers']}")
    print(f"  BLEU                  : {summary['bleu']:.2f}")
    print(f"  Numeric Exact Match   : {summary['num_em']:.4f}  ({summary['num_em']*100:.1f}%)")
    print(f"  Hallucination Rate    : {summary['halluc_rate']:.4f}  ({summary['halluc_rate']*100:.1f}%)")
    print(f"  Count Match Rate      : {summary['count_match']:.4f}  ({summary['count_match']*100:.1f}%)")
    print(f"  Numeric MAE           : {summary['num_mae']:.4f}")
    print(f"  Numeric MAPE          : {summary['num_mape_pct']:.2f}%")
    print(f"  Scalar MSE (log-norm) : {summary['scalar_mse_lognorm']:.6f}  [xVal only]")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model":      "xval",
            "checkpoint": args.checkpoint,
            "summary":    summary,
            "examples":   per_example_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--config",        required=True,  help="model_config.yaml")
    parser.add_argument("--train_config",  required=True,  help="train_config.yaml")
    parser.add_argument("--output",        default="results/xval_results.json")
    parser.add_argument("--max_examples",  type=int, default=7700)
    parser.add_argument("--max_new_tokens",type=int, default=80)
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
