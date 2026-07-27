"""
train/train_bigbench.py
========================
Unified training entry point for ALL 4 models on the BigBench Arithmetic task.

Selects the correct model, tokenizer, dataset, and step_fn based on --model_type.
Everything else (DDP, AMP, early stopping, checkpointing) is handled by Trainer.

Run with torchrun:
  # BPE on BigBench
  torchrun --nproc_per_node=4 train/train_bigbench.py \\
      --model_type bpe \\
      --config configs/bigbench_train_config.yaml \\
      --model_config configs/model_config.yaml

  # xVal on BigBench
  torchrun --nproc_per_node=4 train/train_bigbench.py \\
      --model_type xval \\
      --config configs/bigbench_train_config.yaml \\
      --model_config configs/model_config.yaml

  # FoNE on BigBench
  torchrun --nproc_per_node=4 train/train_bigbench.py \\
      --model_type fone \\
      --config configs/bigbench_train_config.yaml \\
      --model_config configs/model_config.yaml

  # Digit on BigBench  (trains digit tokenizer first if not already done)
  torchrun --nproc_per_node=4 train/train_bigbench.py \\
      --model_type digit \\
      --config configs/bigbench_train_config.yaml \\
      --model_config configs/model_config.yaml
"""

import os
import argparse
import torch
import torch.distributed as dist
import yaml
from tokenizers import Tokenizer

from model.gpt2 import GPT2Config
from model.gpt2_bpe import GPT2BPE
from model.gpt2_xval import GPT2XVal
from model.gpt2_fone import GPT2FoNE
from model.gpt2_digit import GPT2Digit

from data.dataset_bigbench import BigBenchArithmeticDataset, bigbench_collate_fn
from train.trainer import Trainer, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Step functions (one per model type)
# These are the ONLY parts that differ between model types.
# All other training logic (DDP, AMP, checkpointing) is in trainer.py.
# ─────────────────────────────────────────────────────────────────────────────

def bpe_step(model, batch, device):
    """CE-only step for BPE and Digit (no special number tensors needed)."""
    input_ids = batch["input_ids"].to(device)
    labels    = batch["labels"].to(device)
    loss, _   = model(input_ids, labels=labels)
    return loss, {"ce": loss.item()}


def xval_step(model, batch, device):
    """CE + MSE step for xVal."""
    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    number_values  = batch["number_values"].to(device)
    number_mask    = batch["number_mask"].to(device)
    loss, log_dict = model(input_ids, number_values, number_mask, labels=labels)
    return loss, log_dict


def fone_step(model, batch, device):
    """CE-only step for FoNE (number_values used on input side, not as targets)."""
    input_ids     = batch["input_ids"].to(device)
    labels        = batch["labels"].to(device)
    number_values = batch["number_values"].to(device)
    number_mask   = batch["number_mask"].to(device)
    loss, _       = model(input_ids, number_values, number_mask, labels=labels)
    return loss, {"ce": loss.item()}


# Digit is architecturally identical to BPE — same step function
digit_step = bpe_step


# ─────────────────────────────────────────────────────────────────────────────
# Sample function (qualitative progress check at end of each epoch)
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_fn(model_type: str, tokenizer, dev_dataset,
                   device, model_cfg: dict):
    """
    Returns a sample_fn(model_module, epoch) -> None closure.

    Picks the first dev example, feeds the question as a prompt,
    generates the answer, and prints REF vs GEN for progress tracking.
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")
    num_id = tokenizer.token_to_id("[NUM]")

    first_example = dev_dataset[0]
    target        = first_example["target"]

    def sample_fn(model_module, epoch):
        with torch.no_grad():
            # All model types have "input_ids" in the dataset
            input_ids_full = first_example["input_ids"]  # (T,)
            ids_list       = input_ids_full.tolist()

            try:
                sep_pos = ids_list.index(sep_id)
            except ValueError:
                sep_pos = len(ids_list) - 1

            # Build prompt kwargs for each model type
            prompt_ids = input_ids_full[:sep_pos + 1].unsqueeze(0).to(device)

            if model_type in ["bpe", "digit"]:
                output = model_module.generate(
                    prompt_ids, max_new_tokens=40, eos_token_id=eos_id, temperature=0.8
                )
                gen_ids = output[0, sep_pos + 1:].tolist()
                if eos_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_id)]
                generated = tokenizer.decode(gen_ids)

            elif model_type == "xval":
                nv = first_example["number_values"][:sep_pos + 1].unsqueeze(0).to(device)
                nm = first_example["number_mask"][:sep_pos + 1].unsqueeze(0).to(device)
                gen_ids, pred_vals = model_module.generate(
                    prompt_ids, nv, nm, max_new_tokens=40,
                    eos_token_id=eos_id, temperature=0.8
                )
                from data.dataset_xval import denormalize_number
                raw_ids = gen_ids[0, sep_pos + 1:].tolist()
                if eos_id in raw_ids:
                    raw_ids = raw_ids[:raw_ids.index(eos_id)]
                v_iter = iter(pred_vals[0])
                parts, buf = [], []
                for tid in raw_ids:
                    if tid == num_id:
                        if buf: parts.append(tokenizer.decode(buf)); buf = []
                        parts.append(f"{denormalize_number(next(v_iter, 0.0)):.4g}")
                    else:
                        buf.append(tid)
                if buf: parts.append(tokenizer.decode(buf))
                generated = " ".join(parts)

            elif model_type == "fone":
                nv = first_example["number_values"][:sep_pos + 1].unsqueeze(0).to(device)
                nm = first_example["number_mask"][:sep_pos + 1].unsqueeze(0).to(device)
                gen_ids, pred_vals = model_module.generate(
                    prompt_ids, nv, nm, max_new_tokens=40,
                    eos_token_id=eos_id, temperature=0.8
                )
                raw_ids = gen_ids[0, sep_pos + 1:].tolist()
                if eos_id in raw_ids:
                    raw_ids = raw_ids[:raw_ids.index(eos_id)]
                v_iter = iter(pred_vals[0])
                parts, buf = [], []
                for tid in raw_ids:
                    if tid == num_id:
                        if buf: parts.append(tokenizer.decode(buf)); buf = []
                        parts.append(f"{next(v_iter, 0.0):.4g}")
                    else:
                        buf.append(tid)
                if buf: parts.append(tokenizer.decode(buf))
                generated = " ".join(parts)

        print(f"\n{'='*60}")
        print(f"[SAMPLE] End of Epoch {epoch}  (BigBench/{model_type})")
        print(f"  Q   : {first_example.get('example_id', '')}")
        print(f"  REF : {target}")
        print(f"  GEN : {generated}")
        print(f"{'='*60}\n")

    return sample_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    parser = argparse.ArgumentParser(
        description="Train any of the 4 number embedding models on BigBench Arithmetic."
    )
    parser.add_argument("--model_type",   required=True,
                        choices=["bpe", "xval", "fone", "digit"])
    parser.add_argument("--config",       required=True,
                        help="Path to bigbench_train_config.yaml")
    parser.add_argument("--model_config", required=True,
                        help="Path to model_config.yaml")
    parser.add_argument("--resume",       default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    torch.manual_seed(train_cfg["seed"] + rank)
    torch.cuda.manual_seed(train_cfg["seed"] + rank)

    # ── Select tokenizer ───────────────────────────────────────────────────────
    if args.model_type == "digit":
        tok_path = train_cfg.get("digit_tokenizer", "tokenizer/totto_digit/tokenizer.json")
    else:
        tok_path = train_cfg["tokenizer"]
    tokenizer = Tokenizer.from_file(tok_path)

    # ── Load datasets (model_type-aware) ──────────────────────────────────────
    max_len = (train_cfg.get("digit_max_seq_len", 200)
               if args.model_type == "digit"
               else train_cfg["max_seq_len"])

    train_dataset = BigBenchArithmeticDataset(
        data_path   = train_cfg["train_data"],
        tokenizer   = tokenizer,
        model_type  = args.model_type,
        max_seq_len = max_len,
    )
    dev_dataset = BigBenchArithmeticDataset(
        data_path   = train_cfg["dev_data"],
        tokenizer   = tokenizer,
        model_type  = args.model_type,
        max_seq_len = max_len,
    )

    # ── Build model ────────────────────────────────────────────────────────────
    cfg = GPT2Config(**{k: v for k, v in model_cfg.items()
                        if k in GPT2Config.__dataclass_fields__})

    if args.model_type == "bpe":
        model   = GPT2BPE(cfg)
        step_fn = bpe_step
    elif args.model_type == "digit":
        cfg.vocab_size = tokenizer.get_vocab_size()   # digit tokenizer vocab
        model   = GPT2Digit(cfg)
        step_fn = digit_step
    elif args.model_type == "xval":
        model   = GPT2XVal(cfg, num_token_id=model_cfg.get("num_token_id", 8))
        step_fn = xval_step
    elif args.model_type == "fone":
        model   = GPT2FoNE(cfg,
                            num_token_id=model_cfg.get("num_token_id", 8),
                            int_digit_len=model_cfg.get("fone_int_digits", 7),
                            frac_digit_len=model_cfg.get("fone_frac_digits", 2))
        step_fn = fone_step

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"BigBench Arithmetic Training — {args.model_type.upper()}")
        print(f"{'='*60}")
        print(f"  Parameters : {model.num_parameters()/1e6:.2f}M")
        print(f"  Train size : {len(train_dataset):,}")
        print(f"  Dev size   : {len(dev_dataset):,}")
        print(f"  Max seq    : {max_len}")
        print(f"{'='*60}\n")

    if args.resume:
        device = torch.device("cuda", rank)
        load_checkpoint(args.resume, model, device=device)
        if rank == 0:
            print(f"Resumed from: {args.resume}")

    # Run name includes model_type so checkpoints don't collide with ToTTo runs
    run_name = f"{args.model_type}_bigbench"

    trainer = Trainer(
        model         = model,
        train_dataset = train_dataset,
        dev_dataset   = dev_dataset,
        config        = train_cfg,
        rank          = rank,
        world_size    = world_size,
        run_name      = run_name,
    )

    trainer.train(
        step_fn    = step_fn,
        collate_fn = bigbench_collate_fn,
        sample_fn  = make_sample_fn(
            model_type  = args.model_type,
            tokenizer   = tokenizer,
            dev_dataset = dev_dataset,
            device      = torch.device("cuda", rank),
            model_cfg   = model_cfg,
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
