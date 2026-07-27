"""
train/trainer.py
================
Shared training loop used by both train_bpe.py and train_xval.py.

Contains:
  - get_lr()         : cosine decay with linear warmup schedule
  - save_checkpoint(): save model + optimizer state (rank-0 only)
  - load_checkpoint(): resume training from a checkpoint
  - Trainer class   : the full training loop (DDP, AMP, grad accumulation)

Both training scripts call Trainer.train() with slightly different arguments
(the model forward call differs). The loop logic is identical.
"""

import os
import math
import time
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ── Logging setup ─────────────────────────────────────────────────────────────
def get_logger(rank: int, log_path: str = None) -> logging.Logger:
    """
    Create a logger that only prints on rank 0.
    This prevents all 4 GPUs from spamming the same output.
    """
    logger = logging.getLogger(f"trainer.rank{rank}")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (rank 0 only)
    if rank == 0 and log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(
    step:          int,
    warmup_steps:  int,
    total_steps:   int,
    lr_max:        float,
    lr_min:        float,
) -> float:
    """
    Cosine decay with linear warmup.

    Phase 1 (0 → warmup_steps):    LR rises linearly from 0 to lr_max.
    Phase 2 (warmup → total_steps): LR decays via cosine curve to lr_min.

    This is the standard schedule used in GPT-2 and most modern LLMs.

    Example with lr_max=3e-4, lr_min=3e-5, warmup=500, total=50000:
      step=0     → lr=0
      step=500   → lr=3e-4  (peak)
      step=25000 → lr=1.65e-4  (midway through cosine decay)
      step=50000 → lr=3e-5  (final)
    """
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / max(warmup_steps, 1)
    if step >= total_steps:
        return lr_min
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_dir: Path,
    model:          torch.nn.Module,  # the DDP-wrapped model
    optimizer:      torch.optim.Optimizer,
    scaler:         GradScaler,
    epoch:          int,
    global_step:    int,
    config:         dict,
    prefix:         str = "checkpoint",
    keep_last_n:    int = 3,
) -> None:
    """
    Save model, optimizer, and scaler state to disk.

    Only call this on rank 0. Other ranks should not save checkpoints
    (they hold the same weights after DDP gradient sync, so rank 0 is sufficient).

    Parameters
    ----------
    model       : DDP-wrapped model — use model.module to get the actual weights
    keep_last_n : delete older checkpoints to save disk space
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}_epoch{epoch:02d}_step{global_step:06d}.pt"
    filepath = checkpoint_dir / filename

    torch.save(
        {
            "epoch":      epoch,
            "step":       global_step,
            "model":      model.module.state_dict(),   # unwrap DDP
            "optimizer":  optimizer.state_dict(),
            "scaler":     scaler.state_dict(),
            "config":     config,
        },
        filepath,
    )

    # Clean up old checkpoints (keep only the last N)
    existing = sorted(checkpoint_dir.glob(f"{prefix}_epoch*.pt"))
    for old in existing[:-keep_last_n]:
        old.unlink()

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model:           torch.nn.Module,
    optimizer:       torch.optim.Optimizer = None,
    scaler:          GradScaler = None,
    device:          torch.device = None,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer/scaler) state.

    Returns the checkpoint dict (contains 'epoch', 'step', 'config').
    """
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Handles the full training loop for both BPE and xVal models.

    Supports:
      - Multi-GPU via DistributedDataParallel (DDP)
      - Mixed precision (fp16) via torch.cuda.amp
      - Gradient accumulation
      - Cosine LR decay with linear warmup
      - Periodic dev evaluation
      - Checkpointing with cleanup

    The only thing that differs between the BPE and xVal training runs
    is the `step_fn` argument to train(): a callable that performs one
    forward pass and returns the loss (and optionally loss components).
    """

    def __init__(
        self,
        model:         torch.nn.Module,
        train_dataset,
        dev_dataset,
        config:        dict,          # from train_config.yaml
        rank:          int,           # this process's GPU rank (0–3)
        world_size:    int,           # total number of GPUs (4)
        run_name:      str = "run",   # "bpe" or "xval"
    ):
        self.config     = config
        self.rank       = rank
        self.world_size = world_size
        self.run_name   = run_name
        self.device     = torch.device("cuda", rank)

        # ── Logger ────────────────────────────────────────────────────────────
        log_file = os.path.join(config["log_dir"], f"{run_name}_train.log")
        self.log = get_logger(rank, log_file)

        # ── DataLoaders ───────────────────────────────────────────────────────
        # DistributedSampler ensures each GPU gets a disjoint subset of data.
        # set_epoch() must be called each epoch so shuffling differs per epoch.
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config["seed"],
        )
        self.dev_sampler = DistributedSampler(
            dev_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        # Both loaders need a collate_fn — passed in from the training script
        # (bpe_collate_fn or xval_collate_fn). Stored on self in train().
        self._train_dataset = train_dataset
        self._dev_dataset   = dev_dataset

        # ── Model → GPU → DDP ─────────────────────────────────────────────────
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[rank], output_device=rank)

        # ── Optimizer ─────────────────────────────────────────────────────────
        # AdamW with decoupled weight decay.
        # Do NOT apply weight decay to biases and LayerNorm parameters.
        decay_params     = [p for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() >= 2]
        no_decay_params  = [p for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() < 2]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": config["weight_decay"]},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"]),
        )

        # ── AMP Scaler ────────────────────────────────────────────────────────
        # GradScaler prevents fp16 underflow by scaling gradients up before
        # the backward pass, then scaling back before the optimizer step.
        self.scaler = GradScaler(enabled=config.get("use_amp", True))

        # ── Computed training parameters ──────────────────────────────────────
        # Total optimization steps (for LR schedule)
        steps_per_epoch = math.ceil(
            len(train_dataset) / (config["micro_batch_size"] * world_size)
        )
        self.steps_per_epoch = steps_per_epoch
        self.total_steps     = steps_per_epoch * config["max_epochs"]

        if rank == 0:
            self.log.info(f"Run name       : {run_name}")
            self.log.info(f"Train examples : {len(train_dataset):,}")
            self.log.info(f"Dev examples   : {len(dev_dataset):,}")
            self.log.info(f"Steps/epoch    : {steps_per_epoch:,}")
            self.log.info(f"Total steps    : {self.total_steps:,}")
            self.log.info(f"Effective batch: "
                          f"{config['micro_batch_size']} × {world_size} GPUs "
                          f"× {config['gradient_accumulation_steps']} accum "
                          f"= {config['micro_batch_size'] * world_size * config['gradient_accumulation_steps']}")

    def train(self, step_fn, collate_fn, dev_step_fn=None, sample_fn=None):
        """
        Run the full training loop.

        Parameters
        ----------
        step_fn      : callable(model, batch, device) → (loss, log_dict)
                       Performs one forward pass and returns the loss tensor
                       and a dict of scalars to log (e.g. {"ce": 8.2, "mse": 0.5}).
                       This is the ONLY part that differs between BPE and xVal.

        collate_fn   : callable — the DataLoader collate function
                       (bpe_collate_fn or xval_collate_fn)

        dev_step_fn  : callable(model, batch, device) → float (optional)
                       Like step_fn but returns only a loss float, no gradients.
                       If None, dev_step_fn defaults to step_fn with torch.no_grad.

        sample_fn    : callable(model_module, epoch) → None (optional)
                       Called on rank 0 at the END of each epoch with the unwrapped
                       model (model.module) in eval mode. Use this to generate a
                       sample output and print it to the console for progress tracking.
                       Other ranks will simply wait at dist.barrier() — no DDP risk.
        """
        cfg   = self.config
        rank  = self.rank
        accum = cfg["gradient_accumulation_steps"]

        # ── Build DataLoaders ──────────────────────────────────────────────────
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=cfg["micro_batch_size"],
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            drop_last=True,   # drop incomplete last batch for DDP stability
        )
        dev_loader = DataLoader(
            self._dev_dataset,
            batch_size=cfg["micro_batch_size"],
            sampler=self.dev_sampler,
            collate_fn=collate_fn,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
        )

        global_step = 0
        checkpoint_dir = Path(cfg["checkpoint_dir"]) / self.run_name

        # Early stopping initializations
        best_dev_loss = float("inf")
        patience_counter = 0
        patience = cfg.get("early_stopping_patience", 5)
        early_stop = False

        for epoch in range(1, cfg["max_epochs"] + 1):
            # Must call set_epoch() so DistributedSampler shuffles differently
            self.train_sampler.set_epoch(epoch)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_start = time.time()
            running_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):

                # ── Update learning rate ───────────────────────────────────────
                lr = get_lr(
                    step=global_step,
                    warmup_steps=cfg["warmup_steps"],
                    total_steps=self.total_steps,
                    lr_max=cfg["learning_rate"],
                    lr_min=cfg["lr_min"],
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # ── Forward + backward pass ────────────────────────────────────
                # autocast: runs the forward pass in fp16 for speed
                # The GradScaler handles gradient scaling to prevent underflow.
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=cfg.get("use_amp", True)):
                    loss, log_dict = step_fn(self.model, batch, self.device)
                    loss = loss / accum   # scale loss for gradient accumulation

                self.scaler.scale(loss).backward()

                # ── Optimizer step (every `accum` batches) ─────────────────────
                if (batch_idx + 1) % accum == 0:
                    # Unscale before clipping so clip threshold is in real units
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg["grad_clip"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    running_loss += loss.item() * accum

                    # ── Logging (rank 0 only, every 50 steps) ──────────────────
                    if rank == 0 and global_step % 50 == 0:
                        avg_loss = running_loss / 50
                        running_loss = 0.0
                        log_str = " | ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
                        self.log.info(
                            f"epoch={epoch} step={global_step:05d} "
                            f"lr={lr:.2e} | {log_str} | loss_avg={avg_loss:.4f}"
                        )

                    # ── Dev evaluation ─────────────────────────────────────────
                    if global_step % cfg["eval_every_steps"] == 0:
                        dev_loss = self._eval(dev_loader, step_fn, cfg["eval_max_batches"])
                        if rank == 0:
                            self.log.info(
                                f"  [DEV] epoch={epoch} step={global_step} "
                                f"dev_loss={dev_loss:.4f}"
                            )
                        
                        # Early Stopping & Best Checkpoint Saving
                        if dev_loss < best_dev_loss:
                            best_dev_loss = dev_loss
                            patience_counter = 0
                            if rank == 0:
                                best_ckpt_path = save_checkpoint(
                                    checkpoint_dir=checkpoint_dir,
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scaler=self.scaler,
                                    epoch=epoch,
                                    global_step=global_step,
                                    config=cfg,
                                    prefix=f"{self.run_name}_best",
                                    keep_last_n=1,
                                )
                                self.log.info(f"New best dev loss! Saved checkpoint: {best_ckpt_path}")
                        else:
                            patience_counter += 1
                            if rank == 0:
                                self.log.info(
                                    f"No improvement in dev loss. Patience: {patience_counter}/{patience}"
                                )
                        
                        if patience_counter >= patience:
                            if rank == 0:
                                self.log.info(f"Early stopping triggered at step {global_step}. Convergence reached.")
                            early_stop = True
                        
                        self.model.train()

                if early_stop:
                    break
            
            if early_stop:
                break

            # ── End of epoch ──────────────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            if rank == 0:
                self.log.info(
                    f"Epoch {epoch} done in {epoch_time/60:.1f}m | "
                    f"step={global_step}"
                )

            # ── Sample generation (rank 0 only) ───────────────────────────────
            # Switches to eval mode, runs generate() on one dev example, then
            # switches back to train mode. Ranks 1-3 wait at dist.barrier() below.
            if rank == 0 and sample_fn is not None:
                self.model.eval()
                try:
                    with torch.no_grad():
                        sample_fn(self.model.module, epoch)
                except Exception as e:
                    self.log.warning(f"sample_fn failed at epoch {epoch}: {e}")
                self.model.train()

            # ── Checkpoint ────────────────────────────────────────────────────
            if rank == 0 and epoch % cfg["save_every_epochs"] == 0:
                ckpt_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=global_step,
                    config=cfg,
                    prefix=self.run_name,
                    keep_last_n=cfg.get("keep_last_n", 3),
                )
                self.log.info(f"Checkpoint saved: {ckpt_path}")

            # All ranks must reach this barrier before starting the next epoch
            dist.barrier()

        if rank == 0:
            self.log.info("Training complete.")

    @torch.no_grad()
    def _eval(self, dev_loader, step_fn, max_batches: int) -> float:
        """Run the model on the dev set and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in dev_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=self.config.get("use_amp", True)):
                loss, _ = step_fn(self.model, batch, self.device)
            total_loss += loss.item()
            n_batches  += 1
            if n_batches >= max_batches:
                break

        # Average loss across all ranks (so the reported value is consistent)
        loss_tensor = torch.tensor(total_loss / max(n_batches, 1),
                                   device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()
