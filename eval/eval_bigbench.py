"""
eval/eval_bigbench.py
======================
Evaluate any of the 4 models (BPE, xVal, FoNE, Digit) on the
BigBench Arithmetic test set.

For each example:
  1. The input question is fed as a prompt (input + [SEP])
  2. The model generates up to 40 tokens
  3. The first number in the generated output is extracted as the answer
  4. Exact match is computed: does the predicted number == correct answer?

Metric: Exact Numerical Accuracy (ENA) — the fraction of questions where
the extracted predicted number exactly matches the correct target number.
We also report ±1% tolerance accuracy (near-match) for robustness.

Usage:
  # BPE
  python eval/eval_bigbench.py \\
      --model_type bpe \\
      --checkpoint checkpoints/bpe_bigbench/bpe_best.pt \\
      --config configs/model_config.yaml \\
      --train_config configs/bigbench_train_config.yaml \\
      --test_data data/bigbench_arithmetic/test.jsonl \\
      --output results/bigbench/bpe.json

  # xVal
  python eval/eval_bigbench.py \\
      --model_type xval \\
      --checkpoint checkpoints/xval_bigbench/xval_best.pt \\
      ...

  # FoNE
  python eval/eval_bigbench.py --model_type fone ...

  # Digit
  python eval/eval_bigbench.py --model_type digit ...
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from tokenizers import Tokenizer

from model.gpt2 import GPT2Config
from model.gpt2_bpe import GPT2BPE
from model.gpt2_xval import GPT2XVal
from model.gpt2_fone import GPT2FoNE
from model.gpt2_digit import GPT2Digit
from data.dataset_xval import replace_numbers_with_num_token, denormalize_number
from data.dataset_fone import replace_numbers_fone
from tokenizer.train_digit_tokenizer import split_numbers_to_digits
from eval.metrics import extract_numbers


# ── Number comparison helpers ─────────────────────────────────────────────────

def exact_match_number(target_str: str, generated_str: str) -> bool:
    """
    Return True if the first number extracted from `generated_str` matches
    the target number (parsed from `target_str`) exactly.
    """
    target_nums = extract_numbers(target_str)
    if not target_nums:
        return False
    target = target_nums[0]

    gen_nums = extract_numbers(generated_str)
    if not gen_nums:
        return False
    predicted = gen_nums[0]

    # Allow for floating-point equality with a tiny epsilon
    if target == 0.0:
        return abs(predicted) < 1e-6
    return abs(predicted - target) / (abs(target) + 1e-9) < 1e-6


def near_match_number(target_str: str, generated_str: str, tol: float = 0.01) -> bool:
    """
    Return True if the predicted number is within `tol` fraction of the target.
    (±1% tolerance by default.)
    """
    target_nums = extract_numbers(target_str)
    if not target_nums:
        return False
    target = target_nums[0]

    gen_nums = extract_numbers(generated_str)
    if not gen_nums:
        return False
    predicted = gen_nums[0]

    if target == 0.0:
        return abs(predicted) < 1e-6
    return abs(predicted - target) / (abs(target) + 1e-9) <= tol


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt_bpe(inp: str, tokenizer: Tokenizer) -> dict:
    """BPE: just tokenize the question + [SEP]."""
    text = inp + " [SEP]"
    enc  = tokenizer.encode(text)
    return {"input_ids": torch.tensor([enc.ids], dtype=torch.long)}


def build_prompt_digit(inp: str, tokenizer: Tokenizer) -> dict:
    """Digit: split number chars before tokenizing."""
    text = split_numbers_to_digits(inp + " [SEP]")
    enc  = tokenizer.encode(text)
    return {"input_ids": torch.tensor([enc.ids], dtype=torch.long)}


def build_prompt_xval(inp: str, tokenizer: Tokenizer, num_id: int) -> dict:
    """xVal: replace numbers with [NUM], build number_values / number_mask."""
    text, num_vals = replace_numbers_with_num_token(inp + " [SEP]")
    enc  = tokenizer.encode(text)
    ids  = enc.ids

    v_iter = iter(num_vals)
    vals, mask = [], []
    for tid in ids:
        if tid == num_id:
            vals.append(float(next(v_iter, 0.0)))
            mask.append(True)
        else:
            vals.append(0.0)
            mask.append(False)

    return {
        "input_ids":     torch.tensor([ids],  dtype=torch.long),
        "number_values": torch.tensor([vals], dtype=torch.float32),
        "number_mask":   torch.tensor([mask], dtype=torch.bool),
    }


def build_prompt_fone(inp: str, tokenizer: Tokenizer, num_id: int) -> dict:
    """FoNE: same as xVal but with raw (un-normalized) values."""
    text, num_vals = replace_numbers_fone(inp + " [SEP]")
    enc  = tokenizer.encode(text)
    ids  = enc.ids

    v_iter = iter(num_vals)
    vals, mask = [], []
    for tid in ids:
        if tid == num_id:
            vals.append(float(next(v_iter, 0.0)))
            mask.append(True)
        else:
            vals.append(0.0)
            mask.append(False)

    return {
        "input_ids":     torch.tensor([ids],  dtype=torch.long),
        "number_values": torch.tensor([vals], dtype=torch.float32),
        "number_mask":   torch.tensor([mask], dtype=torch.bool),
    }


# ── Output decoders ───────────────────────────────────────────────────────────

def rejoin_digit_tokens(text: str) -> str:
    """
    Reverse the digit-splitting preprocessing applied at training time.

    The digit tokenizer (Metaspace BPE) guarantees that each digit 0-9 is its
    own token — digits are never merged by BPE.  However, the MetaspaceDecoder
    may attach trailing punctuation to the last digit in a sequence, producing
    chunks like "4." or "2," rather than "4" + ".".

    This function handles both cases:
      1. Pure digit chunks: "1", "8", "7", "4" → buffered and joined → "1874"
      2. Digit+punctuation chunks: "4." → digit part "4" buffered, suffix "."
         re-attached after the number is joined → "1874."

    Examples:
      "8 4"        → "84"
      "3 1 0 8"    → "3108"
      "1 8 7 4."   → "1874."
      "2 0 1 6,"   → "2016,"
      "What is 8 4 plus 3?" → "What is 84 plus 3?"
    """
    import string
    PUNCT = string.punctuation

    parts = text.split(" ")
    out   = []
    buf   = []           # accumulates consecutive digit-only chunks

    def flush_buf(suffix=""):
        if buf:
            out.append("".join(buf) + suffix)
            buf.clear()
        elif suffix:
            out.append(suffix)

    for part in parts:
        if not part:
            continue
        core   = part.rstrip(PUNCT)
        suffix = part[len(core):]
        if core.isdigit():
            if suffix:
                buf.append(core)
                flush_buf(suffix)
            else:
                buf.append(core)
        else:
            flush_buf()
            out.append(part)

    flush_buf()
    return " ".join(out)


def decode_bpe_digit(output, prompt_len: int, tokenizer: Tokenizer, eos_id: int) -> str:
    """Decode BPE output (output is a single tensor)."""
    gen_ids = output[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids)


def decode_digit(output, prompt_len: int, tokenizer: Tokenizer, eos_id: int) -> str:
    """
    Decode Digit model output.

    Step 1: Standard token decoding (produces "8 4" for the number 84).
    Step 2: rejoin_digit_tokens() collapses space-separated digits back into
            complete numbers ("8 4" → "84").
    """
    gen_ids = output[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    raw_text = tokenizer.decode(gen_ids)
    return rejoin_digit_tokens(raw_text)


def decode_xval(output, prompt_len: int, tokenizer: Tokenizer,
                eos_id: int, num_id: int) -> str:
    """Decode xVal output: denormalize [NUM] values from log-space."""
    gen_ids, pred_values = output
    gen_ids = gen_ids[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    v_iter = iter(pred_values[0])
    parts, buf = [], []
    for tid in gen_ids:
        if tid == num_id:
            if buf:
                parts.append(tokenizer.decode(buf)); buf = []
            raw_val = next(v_iter, 0.0)
            # xVal predicts in log-normalized space → denormalize before display
            val = denormalize_number(raw_val)
            parts.append(f"{val:.4g}")
        else:
            buf.append(tid)
    if buf:
        parts.append(tokenizer.decode(buf))
    return "".join(parts)


def decode_fone(output, prompt_len: int, tokenizer: Tokenizer,
                eos_id: int, num_id: int) -> str:
    """Decode FoNE output: values are already raw (no denormalization needed)."""
    gen_ids, pred_values = output
    gen_ids = gen_ids[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    reconstructed_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    for val in pred_values[0]:
        reconstructed_text = reconstructed_text.replace("[NUM]", f"{val:.4g}", 1)
    return reconstructed_text.replace("[EOS]", "").strip()


# ── Main evaluation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    # Load tokenizer (digit model uses separate tokenizer)
    if args.model_type == "digit":
        tok_path = train_cfg.get("digit_tokenizer", "tokenizer/totto_digit/tokenizer.json")
    else:
        tok_path = train_cfg["tokenizer"]
    tokenizer = Tokenizer.from_file(tok_path)
    eos_id    = tokenizer.token_to_id("[EOS]")
    num_id    = tokenizer.token_to_id("[NUM]")   # None for digit model

    print(f"Tokenizer: {tok_path}  (vocab={tokenizer.get_vocab_size():,})")

    # Instantiate model
    cfg = GPT2Config(**{k: v for k, v in model_cfg.items()
                        if k in GPT2Config.__dataclass_fields__})
    if args.model_type == "digit":
        cfg.vocab_size = tokenizer.get_vocab_size()

    if args.model_type == "bpe":
        model = GPT2BPE(cfg)
    elif args.model_type == "digit":
        model = GPT2Digit(cfg)
    elif args.model_type == "xval":
        model = GPT2XVal(cfg, num_token_id=model_cfg.get("num_token_id", 8))
    elif args.model_type == "fone":
        model = GPT2FoNE(cfg,
                         num_token_id=model_cfg.get("num_token_id", 8),
                         int_digit_len=model_cfg.get("fone_int_digits", 7),
                         frac_digit_len=model_cfg.get("fone_frac_digits", 2))
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # Load test data
    examples = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"Evaluating {len(examples):,} BigBench Arithmetic examples ...")

    results = []
    exact_count = near_count = 0

    for ex in tqdm(examples, desc=f"BigBench/{args.model_type}"):
        inp    = ex["input"]
        target = ex["target"]

        # Build prompt
        if args.model_type == "bpe":
            prompt_kwargs = build_prompt_bpe(inp, tokenizer)
        elif args.model_type == "digit":
            prompt_kwargs = build_prompt_digit(inp, tokenizer)
        elif args.model_type == "xval":
            prompt_kwargs = build_prompt_xval(inp, tokenizer, num_id)
        elif args.model_type == "fone":
            prompt_kwargs = build_prompt_fone(inp, tokenizer, num_id)

        prompt_len = prompt_kwargs["input_ids"].size(1)
        prompt_kwargs = {k: v.to(device) for k, v in prompt_kwargs.items()}

        # Generate
        output = model.generate(
            **prompt_kwargs,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            temperature=0.0,       # greedy
            repetition_penalty=1.0,
        )

        # Decode
        if args.model_type == "bpe":
            generated = decode_bpe_digit(output, prompt_len, tokenizer, eos_id)
        elif args.model_type == "digit":
            # Digit model generates digits space-separated ("8 4" for 84).
            # decode_digit() joins them back into complete numbers.
            generated = decode_digit(output, prompt_len, tokenizer, eos_id)
        elif args.model_type == "xval":
            generated = decode_xval(output, prompt_len, tokenizer, eos_id, num_id)
        elif args.model_type == "fone":
            generated = decode_fone(output, prompt_len, tokenizer, eos_id, num_id)

        em   = exact_match_number(target, generated)
        near = near_match_number(target, generated, tol=0.01)
        if em:   exact_count += 1
        if near: near_count  += 1

        results.append({
            "id":        ex["id"],
            "input":     inp,
            "target":    target,
            "generated": generated,
            "exact_match":      em,
            "near_match_1pct":  near,
        })

    n = len(examples)
    exact_acc = exact_count / n
    near_acc  = near_count  / n

    print(f"\n{'='*60}")
    print(f"BigBench Arithmetic — {args.model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Total examples         : {n:,}")
    print(f"  Exact Match Accuracy   : {exact_acc*100:.2f}%  ({exact_count}/{n})")
    print(f"  Near Match (±1%) Acc   : {near_acc*100:.2f}%  ({near_count}/{n})")
    print(f"{'='*60}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model_type":    args.model_type,
            "checkpoint":    args.checkpoint,
            "n_examples":    n,
            "exact_accuracy": exact_acc,
            "near_accuracy":  near_acc,
            "results":       results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on BigBench Arithmetic."
    )
    parser.add_argument("--model_type",     required=True,
                        choices=["bpe", "xval", "fone", "digit"])
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--config",         required=True,  help="model_config.yaml")
    parser.add_argument("--train_config",   required=True,  help="bigbench_train_config.yaml")
    parser.add_argument("--test_data",      required=True,
                        help="Path to test.jsonl from download_bigbench.py")
    parser.add_argument("--output",         required=True,
                        help="Where to save the JSON results")
    parser.add_argument("--max_examples",   type=int, default=-1,
                        help="-1 means evaluate all examples (default)")
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Max tokens to generate per answer (default: 40)")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
