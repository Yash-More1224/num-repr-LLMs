"""
train/train_bpe.py
==================
Training entry point for the BASELINE BPE-GPT2 model.

Run with torchrun for multi-GPU DDP training:
  torchrun --nproc_per_node=4 train/train_bpe.py \\
      --config configs/train_config.yaml \\
      --model_config configs/model_config.yaml \\
      --resume checkpoints/bpe/bpe_epoch02_step012000.pt  # optional

This script is intentionally thin — all training logic lives in trainer.py.
The only things specific to BPE here are:
  1. Importing GPT2BPE and ToTToDataset
  2. The `bpe_step()` forward function
"""

import os
import argparse

import torch
import torch.distributed as dist
import yaml

from model.gpt2 import GPT2Config
from model.gpt2_bpe import GPT2BPE
from data.dataset import ToTToDataset, bpe_collate_fn
from tokenizers import Tokenizer
from train.trainer import Trainer, load_checkpoint


# ── Step function (BPE-specific) ──────────────────────────────────────────────

def bpe_step(model, batch, device) -> tuple:
    """
    One forward pass for the BPE model.

    Parameters
    ----------
    model  : DDP-wrapped GPT2BPE
    batch  : dict from bpe_collate_fn
    device : torch.device

    Returns
    -------
    loss     : scalar tensor (autograd-tracked)
    log_dict : dict of floats to log (just "ce" for BPE)
    """
    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    loss, logits = model(input_ids, labels=labels, attention_mask=attention_mask)

    return loss, {"ce": loss.item()}


# ── Sample function (per-epoch qualitative check) ─────────────────────────

def make_bpe_sample_fn(tokenizer, dev_dataset, device):
    """
    Returns a sample_fn(model_module, epoch) -> None closure.

    Grabs the first dev example, builds the prompt (tokens up to and
    including [SEP]), runs model.generate(), and prints the reference
    sentence versus the generated one. Called at the end of every epoch
    on rank 0 so you can visually track learning progress.
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")

    # Pull the first dev record once at startup (no repeated disk I/O)
    first_record = dev_dataset[0]
    reference    = first_record["target"]

    def sample_fn(model_module, epoch):
        input_ids_full = first_record["input_ids"]   # (T,) full sequence
        ids_list = input_ids_full.tolist()

        # Find [SEP] and build prompt (everything up to and including it)
        sep_pos = ids_list.index(sep_id) if sep_id in ids_list else len(ids_list) - 1
        prompt  = input_ids_full[:sep_pos + 1].unsqueeze(0).to(device)  # (1, T_prompt)

        # Generate (greedy decoding)
        output_ids = model_module.generate(
            prompt,
            max_new_tokens=80,
            eos_token_id=eos_id,
            temperature=0.0,
        )

        # Decode the generated portion only
        gen_ids = output_ids[0, sep_pos + 1:].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]
        generated = tokenizer.decode(gen_ids)

        print(f"\n{'='*60}")
        print(f"[SAMPLE] End of Epoch {epoch}")
        print(f"  REF : {reference}")
        print(f"  GEN : {generated}")
        print(f"{'='*60}\n")

    return sample_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── DDP initialization ─────────────────────────────────────────────────────
    # torchrun sets LOCAL_RANK, RANK, and WORLD_SIZE automatically.
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
    torch.manual_seed(train_cfg["seed"] + rank)   # different seed per rank
    torch.cuda.manual_seed(train_cfg["seed"] + rank)

    # ── Tokenizer & datasets ───────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])

    train_dataset = ToTToDataset(
        data_path   = train_cfg["train_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )
    dev_dataset = ToTToDataset(
        data_path   = train_cfg["dev_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg["max_seq_len"],
    )

    # ── Build model ────────────────────────────────────────────────────────────
    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    model = GPT2BPE(config)

    if rank == 0:
        n_params = model.num_parameters()
        print(f"\n{'='*60}")
        print(f"BPE-GPT2 Training")
        print(f"{'='*60}")
        print(f"Model parameters : {n_params/1e6:.2f}M")
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
        run_name      = "bpe",
    )

    trainer.train(
        step_fn    = bpe_step,
        collate_fn = bpe_collate_fn,
        sample_fn  = make_bpe_sample_fn(
            tokenizer   = tokenizer,
            dev_dataset = dev_dataset,
            device      = torch.device("cuda", rank),
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
