"""
eval/eval_fone.py
=================
Evaluate the FoNE-GPT2 model on the ToTTo dev set.

Mirrors eval_bpe.py exactly, with two differences:
  1. Uses GPT2FoNE and ToTToDatasetFoNE (raw number values, number_mask)
  2. At [NUM] positions in the generated output, decodes the number value
     from the FoNE hidden state via the inverse transform (atan2-based decoding)

Usage:
  python eval/eval_fone.py \\
      --checkpoint checkpoints/fone/fone_best_epoch*.pt \\
      --config configs/model_config.yaml \\
      --train_config configs/train_config.yaml \\
      --output results/totto/fone.json
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from tokenizers import Tokenizer

from model.gpt2 import GPT2Config
from model.gpt2_fone import GPT2FoNE
from data.dataset_xval import NUMBER_RE, parse_number
from eval.metrics import (extract_numbers, numeric_exact_match, numeric_mae,
                           numeric_mape, hallucination_rate, aggregate_metrics)


def build_fone_prompt(example: dict, sep_id: int, num_id: int,
                      tokenizer: Tokenizer, max_seq_len: int) -> tuple:
    """
    Build the prompt tensor and associated number_values / number_mask
    for the FoNE model.

    Returns: (prompt_ids (1,T), number_values (1,T), number_mask (1,T))
    """
    from data.dataset_fone import replace_numbers_fone
    linearized = example["linearized"]
    fone_text, raw_values = replace_numbers_fone(linearized)

    encoding  = tokenizer.encode(fone_text)
    token_ids = encoding.ids

    try:
        sep_pos    = token_ids.index(sep_id)
        prompt_ids = token_ids[:sep_pos + 1]
    except ValueError:
        prompt_ids = token_ids

    max_prompt_len = max_seq_len - 80
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    T = len(prompt_ids)
    # Build number_values and number_mask for prompt
    num_iter = iter(raw_values)
    vals     = []
    mask     = []
    for tid in prompt_ids:
        if tid == num_id:
            vals.append(float(next(num_iter, 0.0)))
            mask.append(True)
        else:
            vals.append(0.0)
            mask.append(False)

    prompt_ids_t = torch.tensor([prompt_ids],     dtype=torch.long)
    values_t     = torch.tensor([vals],            dtype=torch.float32)
    mask_t       = torch.tensor([mask],            dtype=torch.bool)
    return prompt_ids_t, values_t, mask_t


@torch.no_grad()
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])
    eos_id    = tokenizer.token_to_id("[EOS]")
    sep_id    = tokenizer.token_to_id("[SEP]")
    num_id    = tokenizer.token_to_id("[NUM]")
    print(f"Tokenizer loaded. EOS={eos_id}, SEP={sep_id}, NUM={num_id}")

    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    model = GPT2FoNE(
        config         = config,
        num_token_id   = model_cfg.get("num_token_id", 8),
        int_digit_len  = model_cfg.get("fone_int_digits", 7),
        frac_digit_len = model_cfg.get("fone_frac_digits", 2),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    data_path = train_cfg["train_data"] if args.split == "train" else train_cfg["dev_data"]
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"Evaluating {len(examples)} {args.split} examples ...")

    per_example_results = []

    for example in tqdm(examples, desc="FoNE eval"):
        reference = example["target"]
        ref_nums  = extract_numbers(reference)

        prompt_ids, values_t, mask_t = build_fone_prompt(
            example, sep_id, num_id, tokenizer, train_cfg["max_seq_len"]
        )
        prompt_len = prompt_ids.size(1)

        gen_ids, pred_values = model.generate(
            prompt_ids.to(device),
            values_t.to(device),
            mask_t.to(device),
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            temperature=0.0,
            repetition_penalty=1.0,
        )

        # Decode generated portion: decode full string with special tokens to preserve spaces around [NUM]
        gen_token_list = gen_ids[0, prompt_len:].tolist()
        if eos_id in gen_token_list:
            gen_token_list = gen_token_list[:gen_token_list.index(eos_id)]

        reconstructed_text = tokenizer.decode(gen_token_list, skip_special_tokens=False)
        for val in pred_values[0]:
            reconstructed_text = reconstructed_text.replace("[NUM]", f"{val:.4g}", 1)
        generated = reconstructed_text.replace("[EOS]", "").strip()

        gen_nums = extract_numbers(generated)

        em     = numeric_exact_match(ref_nums, gen_nums)
        mae    = numeric_mae(ref_nums, gen_nums)
        mape   = numeric_mape(ref_nums, gen_nums)
        halluc = hallucination_rate(ref_nums, gen_nums)

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

    print(f"\n{'='*60}")
    print(f"FoNE-GPT2 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Examples evaluated    : {summary['n_examples']}")
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
        json.dump({"model": "fone", "checkpoint": args.checkpoint,
                   "summary": summary, "examples": per_example_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--config",         required=True,  help="model_config.yaml")
    parser.add_argument("--train_config",   required=True,  help="train_config.yaml")
    parser.add_argument("--output",         default="results/totto/fone.json")
    parser.add_argument("--split",          default="dev", choices=["train", "dev"])
    parser.add_argument("--max_examples",   type=int, default=7700)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
