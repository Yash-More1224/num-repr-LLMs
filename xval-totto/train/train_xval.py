"""
train/train_xval.py
===================
Training entry point for the EXPERIMENTAL xVal-GPT2 model.

Run with torchrun for multi-GPU DDP training:
  torchrun --nproc_per_node=4 train/train_xval.py \\
      --config configs/train_config.yaml \\
      --model_config configs/model_config.yaml \\
      --resume checkpoints/xval/xval_epoch02_step012000.pt  # optional

This script is identical in structure to train_bpe.py.
The only differences from BPE are:
  1. Importing GPT2XVal and ToTToDatasetXVal
  2. The `xval_step()` forward function (passes number_values and number_mask,
     receives ce_loss and mse_loss separately for logging)
"""

import os
import argparse

import torch
import torch.distributed as dist
import yaml

from model.gpt2 import GPT2Config
from model.gpt2_xval import GPT2XVal
from data.dataset_xval import ToTToDatasetXVal, xval_collate_fn
from tokenizers import Tokenizer
from train.trainer import Trainer, load_checkpoint


# ── Step function (xVal-specific) ────────────────────────────────────────────

def xval_step(model, batch, device) -> tuple:
    """
    One forward pass for the xVal model.

    Parameters
    ----------
    model  : DDP-wrapped GPT2XVal
    batch  : dict from xval_collate_fn
    device : torch.device

    Returns
    -------
    loss     : scalar tensor (autograd-tracked)
    log_dict : dict of floats to log ("ce", "mse", "total")
    """
    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    number_values  = batch["number_values"].to(device)
    number_mask    = batch["number_mask"].to(device)
    # attention_mask is in the batch but not needed for the forward pass

    total_loss, logits, ce_loss, mse_loss = model(
        input_ids,
        number_values,
        number_mask,
        labels=labels,
    )

    return total_loss, {
        "ce":    ce_loss.item(),
        "mse":   mse_loss.item(),
        "total": total_loss.item(),
    }


# ── Sample function (per-epoch qualitative check) ─────────────────────────

def make_xval_sample_fn(tokenizer, dev_dataset, device, num_token_id=8):
    """
    Returns a sample_fn(model_module, epoch) -> None closure for xVal.

    Identical in spirit to make_bpe_sample_fn but passes number_values
    and number_mask to the xVal generate() call and splices the predicted
    scalar values back into the decoded string for display.
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")
    num_id = num_token_id

    first_record = dev_dataset[0]
    reference    = first_record["target"]

    def sample_fn(model_module, epoch):
        input_ids_full = first_record["input_ids"]      # (T,)
        num_vals_full  = first_record["number_values"]  # (T,)
        num_mask_full  = first_record["number_mask"]    # (T,)
        ids_list = input_ids_full.tolist()

        sep_pos = ids_list.index(sep_id) if sep_id in ids_list else len(ids_list) - 1

        prompt_ids  = input_ids_full[:sep_pos + 1].unsqueeze(0).to(device)
        prompt_vals = num_vals_full[:sep_pos + 1].unsqueeze(0).to(device)
        prompt_mask = num_mask_full[:sep_pos + 1].unsqueeze(0).to(device)

        gen_ids, gen_scalar_vals = model_module.generate(
            prompt_ids,
            prompt_vals,
            prompt_mask,
            max_new_tokens=80,
            eos_token_id=eos_id,
            temperature=0.8,
        )

        # Decode the generated portion
        new_ids = gen_ids[0, sep_pos + 1:].tolist()
        if eos_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_id)]

        # Splice predicted scalar values back into [NUM] placeholders
        val_idx = 0
        decoded_tokens = []
        for tok in new_ids:
            if tok == num_id and val_idx < len(gen_scalar_vals):
                decoded_tokens.append(f"{gen_scalar_vals[val_idx]:.4g}")
                val_idx += 1
            else:
                decoded_tokens.append(tokenizer.decode([tok]))
        generated = "".join(decoded_tokens)

        print(f"\n{'='*60}")
        print(f"[SAMPLE] End of Epoch {epoch}")
        print(f"  REF : {reference}")
        print(f"  GEN : {generated}")
        print(f"{'='*60}\n")

    return sample_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── DDP initialization ─────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # ── Load configs ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True, help="Path to train_config.yaml")
    parser.add_argument("--model_config", required=True, help="Path to model_config.yaml")
    parser.add_argument("--resume",       default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # ── Reproducibility ────────────────────────────────────────────────────────
    torch.manual_seed(train_cfg["seed"] + rank)
    torch.cuda.manual_seed(train_cfg["seed"] + rank)

    # ── Tokenizer & datasets ───────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])

    train_dataset = ToTToDatasetXVal(
        data_path   = train_cfg["train_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )
    dev_dataset = ToTToDatasetXVal(
        data_path   = train_cfg["dev_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )

    # ── Build model ────────────────────────────────────────────────────────────
    # Filter model_cfg to only the fields GPT2Config accepts
    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    model = GPT2XVal(
        config       = config,
        num_token_id = model_cfg.get("num_token_id", 8),
        mse_lambda   = model_cfg.get("mse_lambda", 1.0),
    )

    if rank == 0:
        n_params = model.num_parameters()
        print(f"\n{'='*60}")
        print(f"xVal-GPT2 Training")
        print(f"{'='*60}")
        print(f"Model parameters : {n_params/1e6:.2f}M")
        print(f"num_token_id     : {model_cfg.get('num_token_id', 8)}")
        print(f"mse_lambda       : {model_cfg.get('mse_lambda', 1.0)}")
        print(f"{'='*60}\n")

    # ── Resume from checkpoint (optional) ─────────────────────────────────────
    if args.resume:
        device = torch.device("cuda", rank)
        load_checkpoint(args.resume, model, device=device)
        if rank == 0:
            print(f"Resumed from: {args.resume}")

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        train_dataset = train_dataset,
        dev_dataset   = dev_dataset,
        config        = train_cfg,
        rank          = rank,
        world_size    = world_size,
        run_name      = "xval",
    )

    trainer.train(
        step_fn    = xval_step,
        collate_fn = xval_collate_fn,
        sample_fn  = make_xval_sample_fn(
            tokenizer    = tokenizer,
            dev_dataset  = dev_dataset,
            device       = torch.device("cuda", rank),
            num_token_id = model_cfg.get("num_token_id", 8),
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
