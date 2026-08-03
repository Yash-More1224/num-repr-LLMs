"""
train/train_fone.py
===================
Training entry point for the FoNE-GPT2 model.

Run with torchrun for multi-GPU DDP training:
  torchrun --nproc_per_node=4 train/train_fone.py \\
      --config configs/train_config.yaml \\
      --model_config configs/model_config.yaml

FoNE specifics vs BPE:
  - Loads ToTToDatasetFoNE instead of ToTToDataset
  - Passes number_values and number_mask to model.forward()
  - CE loss only — no MSE
  - At generation time, [NUM] tokens are decoded via FoNE inverse transform
"""

import os
import argparse

import torch
import torch.distributed as dist
import yaml

from model.gpt2 import GPT2Config
from model.gpt2_fone import GPT2FoNE
from data.dataset_fone import ToTToDatasetFoNE, fone_collate_fn
from data.dataset_xval import denormalize_number   # reuse for any display (not used in FoNE loss)
from tokenizers import Tokenizer
from train.trainer import Trainer, load_checkpoint


# ── Step function (FoNE-specific) ─────────────────────────────────────────────

def fone_step(model, batch, device) -> tuple:
    """
    One forward pass for the FoNE model.

    CE loss only — no MSE (FoNE does not use a regression head).

    Returns
    -------
    loss     : scalar tensor
    log_dict : {"ce": float}
    """
    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    number_values  = batch["number_values"].to(device)
    number_mask    = batch["number_mask"].to(device)

    loss, logits = model(input_ids, number_values, number_mask, labels=labels, target_values=number_values)

    return loss, {"ce": loss.item()}


# ── Sample function ───────────────────────────────────────────────────────────

def make_fone_sample_fn(tokenizer, dev_dataset, device):
    """
    Returns a sample_fn(model_module, epoch) -> None closure.

    Generates one example at the end of each epoch using FoNE decoding.
    [NUM] tokens in output are replaced with their FoNE-decoded values.
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")
    num_id = tokenizer.token_to_id("[NUM]")

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

        # Decode: for [NUM] tokens, substitute with the predicted float
        gen_token_list = gen_ids[0, sep_pos + 1:].tolist()
        if eos_id in gen_token_list:
            gen_token_list = gen_token_list[:gen_token_list.index(eos_id)]

        # Build output string: decode full string with special tokens to preserve spaces around [NUM]
        reconstructed_text = tokenizer.decode(gen_token_list, skip_special_tokens=False)
        for val in pred_values[0]:
            reconstructed_text = reconstructed_text.replace("[NUM]", f"{val:.4g}", 1)
        generated = reconstructed_text.replace("[EOS]", "").strip()

        print(f"\n{'='*60}")
        print(f"[SAMPLE] End of Epoch {epoch}")
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
    parser.add_argument("--config",       required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--resume",       default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    torch.manual_seed(train_cfg["seed"] + rank)
    torch.cuda.manual_seed(train_cfg["seed"] + rank)

    tokenizer = Tokenizer.from_file(train_cfg["tokenizer"])

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

    model = GPT2FoNE(
        config         = config,
        num_token_id   = model_cfg.get("num_token_id", 8),
        int_digit_len  = model_cfg.get("fone_int_digits", 7),
        frac_digit_len = model_cfg.get("fone_frac_digits", 2),
    )

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FoNE-GPT2 Training")
        print(f"{'='*60}")
        print(f"Model parameters : {model.num_parameters()/1e6:.2f}M")
        print(f"FoNE dim         : {model.fone.fone_dim}")
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
        run_name      = "fone",
    )

    trainer.train(
        step_fn    = fone_step,
        collate_fn = fone_collate_fn,
        sample_fn  = make_fone_sample_fn(
            tokenizer   = tokenizer,
            dev_dataset = dev_dataset,
            device      = torch.device("cuda", rank),
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
