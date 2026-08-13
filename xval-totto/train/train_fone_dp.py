"""
train/train_fone_dp.py
======================
Training entry point for the FoNE-DP model (FoNE + Detached Projection).

Run with torchrun for multi-GPU DDP training:
  torchrun --nproc_per_node=4 train/train_fone_dp.py \\
      --config configs/train_config.yaml \\
      --model_config configs/model_config_fone_dp.yaml \\
      --resume checkpoints/fone_dp/fone_dp_epoch02_step012000.pt  # optional

Differences from train_fone.py:
  1. Imports GPT2FoNeDP instead of GPT2FoNE
  2. step_fn logs ce_loss and fone_dp_loss separately
  3. run_name = "fone_dp" (checkpoints saved to checkpoints/fone_dp/)
  4. Uses the same ToTToDatasetFoNE and fone_collate_fn — no data changes
"""

import os
import argparse

import torch
import torch.distributed as dist
import yaml

from model.gpt2 import GPT2Config
from model.gpt2_fone_dp import GPT2FoNeDP
from data.dataset_fone import ToTToDatasetFoNE, fone_collate_fn
from tokenizers import Tokenizer
from train.trainer import Trainer, load_checkpoint


# ── Step function (FoNE-DP-specific) ─────────────────────────────────────────

def fone_dp_step(model, batch, device) -> tuple:
    """
    One forward pass for the FoNE-DP model.

    Unlike the standard FoNE step (which only logs CE loss), we log both
    ce_loss and fone_dp_loss separately to verify that gradient isolation
    is working: ce_loss should behave identically to standard FoNE, while
    fone_dp_loss should converge independently.

    Returns
    -------
    loss     : scalar tensor (autograd-tracked)
    log_dict : {"ce": float, "fone_dp": float, "total": float}
    """
    input_ids     = batch["input_ids"].to(device)
    labels        = batch["labels"].to(device)
    number_values = batch["number_values"].to(device)
    number_mask   = batch["number_mask"].to(device)

    total_loss, logits, ce_loss, fone_dp_loss = model(
        input_ids,
        number_values,
        number_mask,
        labels=labels,
        target_values=number_values,
    )

    return total_loss, {
        "ce":      ce_loss.item(),
        "fone_dp": fone_dp_loss.item(),
        "total":   total_loss.item(),
    }


# ── Sample function (per-epoch qualitative check) ─────────────────────────────

def make_fone_dp_sample_fn(tokenizer, dev_dataset, device):
    """
    Returns a sample_fn(model_module, epoch) -> None closure.

    Generates one dev example at the end of each epoch.
    [NUM] tokens are replaced with values decoded from num_proj → fone.decode().
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")

    first_record = dev_dataset[0]
    reference    = first_record["target"]

    def sample_fn(model_module, epoch):
        input_ids_full = first_record["input_ids"]
        num_vals_full  = first_record["number_values"]
        num_mask_full  = first_record["number_mask"]
        ids_list       = input_ids_full.tolist()

        sep_pos = ids_list.index(sep_id) if sep_id in ids_list else len(ids_list) - 1

        prompt_ids  = input_ids_full[:sep_pos + 1].unsqueeze(0).to(device)
        prompt_vals = num_vals_full[:sep_pos + 1].unsqueeze(0).to(device)
        prompt_mask = num_mask_full[:sep_pos + 1].unsqueeze(0).to(device)

        gen_ids, pred_values = model_module.generate(
            prompt_ids, prompt_vals, prompt_mask,
            max_new_tokens=80,
            eos_token_id=eos_id,
            temperature=0.0,
        )

        gen_token_list = gen_ids[0, sep_pos + 1:].tolist()
        if eos_id in gen_token_list:
            gen_token_list = gen_token_list[:gen_token_list.index(eos_id)]

        reconstructed_text = tokenizer.decode(gen_token_list, skip_special_tokens=False)
        for val in pred_values[0]:
            reconstructed_text = reconstructed_text.replace("[NUM]", f"{val:.4g}", 1)
        generated = reconstructed_text.replace("[EOS]", "").strip()

        print(f"\n{'='*60}")
        print(f"[SAMPLE] End of Epoch {epoch}  (FoNE-DP)")
        print(f"  REF : {reference}")
        print(f"  GEN : {generated}")
        print(f"{'='*60}\n")

    return sample_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True, help="Path to train_config.yaml")
    parser.add_argument("--model_config", required=True, help="Path to model_config_fone_dp.yaml")
    parser.add_argument("--resume",       default=None,  help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    torch.manual_seed(train_cfg["seed"] + rank)
    torch.cuda.manual_seed(train_cfg["seed"] + rank)

    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])

    # Same dataset as standard FoNE — no data changes needed
    train_dataset = ToTToDatasetFoNE(
        data_path   = train_cfg["train_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )
    dev_dataset = ToTToDatasetFoNE(
        data_path   = train_cfg["dev_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )

    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})

    model = GPT2FoNeDP(
        config          = config,
        num_token_id    = model_cfg.get("num_token_id", 8),
        int_digit_len   = model_cfg.get("fone_int_digits", 7),
        frac_digit_len  = model_cfg.get("fone_frac_digits", 2),
        num_proj_hidden = model_cfg.get("num_proj_hidden", 64),
        num_loss_weight = model_cfg.get("num_loss_weight", 1.0),
    )

    if rank == 0:
        proj_params = sum(p.numel() for p in model.num_proj.parameters())
        print(f"\n{'='*60}")
        print(f"FoNE-DP Training")
        print(f"{'='*60}")
        print(f"Model parameters : {model.num_parameters()/1e6:.2f}M")
        print(f"FoNE dim         : {model.fone.fone_dim}")
        print(f"num_proj params  : {proj_params:,}  (detached from backbone)")
        print(f"num_proj_hidden  : {model_cfg.get('num_proj_hidden', 64)}")
        print(f"num_loss_weight  : {model_cfg.get('num_loss_weight', 1.0)}")
        print(f"{'='*60}\n")

    if args.resume:
        device = torch.device("cuda", rank)
        load_checkpoint(args.resume, model, device=device)
        if rank == 0:
            print(f"Resumed from: {args.resume}")

    trainer = Trainer(
        model         = model,
        train_dataset = train_dataset,
        dev_dataset   = dev_dataset,
        config        = train_cfg,
        rank          = rank,
        world_size    = world_size,
        run_name      = "fone_dp",   # checkpoints → checkpoints/fone_dp/
    )

    trainer.train(
        step_fn    = fone_dp_step,
        collate_fn = fone_collate_fn,
        sample_fn  = make_fone_dp_sample_fn(
            tokenizer   = tokenizer,
            dev_dataset = dev_dataset,
            device      = torch.device("cuda", rank),
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
