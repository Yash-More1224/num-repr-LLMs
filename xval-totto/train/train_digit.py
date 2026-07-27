"""
train/train_digit.py
=====================
Training entry point for the Digit Tokenization model.

Run with torchrun for multi-GPU DDP training:
  torchrun --nproc_per_node=4 train/train_digit.py \\
      --config configs/train_config.yaml \\
      --model_config configs/digit_model_config.yaml

Digit-specific vs BPE:
  - Uses totto_digit tokenizer (train_digit_tokenizer.py must be run first)
  - Uses GPT2Digit model (architecturally identical to GPT2BPE)
  - Uses ToTToDatasetDigit (numbers pre-split into digits before tokenization)
  - Uses max_seq_len from config (should be ~512 for digit sequences)
  - Sample function joins digit tokens back into readable numbers at generation time
"""

import os
import argparse

import torch
import torch.distributed as dist
import yaml

from model.gpt2 import GPT2Config
from model.gpt2_digit import GPT2Digit
from data.dataset_digit import ToTToDatasetDigit, digit_collate_fn
from tokenizer.train_digit_tokenizer import split_numbers_to_digits
from tokenizers import Tokenizer
from train.trainer import Trainer, load_checkpoint


# ── Step function (Digit — identical to BPE) ──────────────────────────────────

def digit_step(model, batch, device) -> tuple:
    """
    One forward pass for the Digit model.
    Identical to bpe_step — same CE loss, same interface.
    """
    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    loss, logits = model(input_ids, labels=labels, attention_mask=attention_mask)

    return loss, {"ce": loss.item()}


# ── Sample function ───────────────────────────────────────────────────────────

def make_digit_sample_fn(tokenizer, dev_dataset, device):
    """
    Returns a sample_fn(model_module, epoch) -> None closure.

    Generates text and decodes digit tokens back to readable numbers.
    """
    sep_id = tokenizer.token_to_id("[SEP]")
    eos_id = tokenizer.token_to_id("[EOS]")

    first_record = dev_dataset[0]
    reference    = first_record["target"]

    def sample_fn(model_module, epoch):
        input_ids_full = first_record["input_ids"]
        ids_list       = input_ids_full.tolist()

        sep_pos = ids_list.index(sep_id) if sep_id in ids_list else len(ids_list) - 1
        prompt  = input_ids_full[:sep_pos + 1].unsqueeze(0).to(device)

        output_ids = model_module.generate(
            prompt,
            max_new_tokens=120,   # digit sequences are longer
            eos_token_id=eos_id,
            temperature=0.8,
        )

        gen_ids = output_ids[0, sep_pos + 1:].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]

        # Decode normally — digit tokens will show as individual digits
        # The tokenizer was trained on digit-split text, so decoding
        # will produce strings like "1 8 7 4" for number 1874
        generated = tokenizer.decode(gen_ids)

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

    # Digit tokenizer path can be overridden in train_config or use default
    tokenizer_path = train_cfg.get("digit_tokenizer",
                                   train_cfg.get("tokenizer", "tokenizer/totto_bpe/tokenizer.json"))
    tokenizer = Tokenizer.from_file(tokenizer_path)

    train_dataset = ToTToDatasetDigit(
        data_path   = train_cfg["train_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg.get("digit_max_seq_len", train_cfg["max_seq_len"]),
    )
    dev_dataset = ToTToDatasetDigit(
        data_path   = train_cfg["dev_data"],
        tokenizer   = tokenizer,
        max_seq_len = train_cfg.get("digit_max_seq_len", train_cfg["max_seq_len"]),
    )

    config = GPT2Config(**{k: v for k, v in model_cfg.items()
                           if k in GPT2Config.__dataclass_fields__})
    # Override vocab_size from the digit tokenizer
    config.vocab_size = tokenizer.get_vocab_size()

    model = GPT2Digit(config)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Digit-GPT2 Training")
        print(f"{'='*60}")
        print(f"Model parameters : {model.num_parameters()/1e6:.2f}M")
        print(f"Digit vocab size : {config.vocab_size:,}")
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
        run_name      = "digit",
    )

    trainer.train(
        step_fn    = digit_step,
        collate_fn = digit_collate_fn,
        sample_fn  = make_digit_sample_fn(
            tokenizer   = tokenizer,
            dev_dataset = dev_dataset,
            device      = torch.device("cuda", rank),
        ) if rank == 0 else None,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
