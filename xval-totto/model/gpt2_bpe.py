"""
model/gpt2_bpe.py
=================
The BASELINE model: standard GPT-2 with BPE tokenization.

Numbers are treated like any other text — they are tokenized by BPE and
looked up in the embedding table. No special handling. This is the control
condition in our experiment.

Architecture (identical to GPT-2 small):
  Input tokens
       ↓
  Token Embedding (wte) + Position Embedding (wpe)
       ↓
  GPT2Backbone (12 transformer layers)
       ↓
  LM Head: Linear(768, vocab_size)  [weights tied to wte]
       ↓
  Cross-entropy loss on target tokens only

Usage
-----
  from model.gpt2 import GPT2Config
  from model.gpt2_bpe import GPT2BPE

  config = GPT2Config()
  model  = GPT2BPE(config)
  loss, logits = model(input_ids, labels=labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt2 import GPT2Config, GPT2Backbone, init_weights


class GPT2BPE(nn.Module):
    """
    Standard GPT-2 language model with BPE tokenization (the baseline).

    Parameters
    ----------
    config : GPT2Config
        Architecture hyperparameters. Load from configs/model_config.yaml.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # ── Embedding layers ──────────────────────────────────────────────────
        # wte: token embedding — maps each token ID to a d_model vector
        # wpe: position embedding — maps each position 0..T-1 to a d_model vector
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # ── Shared transformer backbone ───────────────────────────────────────
        # This is IDENTICAL to the backbone in GPT2XVal.
        self.backbone = GPT2Backbone(config)

        # ── Language model head ───────────────────────────────────────────────
        # Projects hidden states to vocabulary logits.
        # Weight tying: lm_head and wte share the SAME weight matrix.
        # This is standard in GPT-2 and saves ~12M parameters for 16K vocab.
        # It also enforces that the model encodes similar tokens similarly
        # in both the input and output spaces.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # ← weight tying

        # ── Weight initialization ─────────────────────────────────────────────
        self.apply(lambda m: init_weights(m, config.n_layer))

    def forward(
        self,
        input_ids:      torch.Tensor,          # (B, T) int64
        labels:         torch.Tensor = None,   # (B, T) int64, -100 where ignored
        attention_mask: torch.Tensor = None,   # (B, T) int64, not used in forward
                                               # (causal mask handles this)
    ) -> tuple:
        """
        Forward pass.

        Parameters
        ----------
        input_ids      : (B, T) — token IDs from the BPE tokenizer
        labels         : (B, T) — same as input_ids but -100 for input portion
                         and padding. If None, loss is not computed.
        attention_mask : (B, T) — padding mask (not used in forward; kept for
                         interface compatibility with DataLoader batches)

        Returns
        -------
        (loss, logits)
          loss   : scalar tensor if labels provided, else None
          logits : (B, T, vocab_size) raw unnormalized scores
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Token + position embeddings ───────────────────────────────────────
        # positions: [0, 1, 2, ..., T-1] — same for every item in the batch
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        x = self.wte(input_ids) + self.wpe(positions)             # (B, T, n_embd)

        # ── Transformer ───────────────────────────────────────────────────────
        hidden = self.backbone(x)   # (B, T, n_embd)

        # ── LM head ───────────────────────────────────────────────────────────
        logits = self.lm_head(hidden)   # (B, T, vocab_size)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Cross-entropy over all (B*T) positions.
            # Positions with labels == -100 are automatically ignored
            # (that is the contract of F.cross_entropy with ignore_index).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                labels.view(-1),                    # (B*T,)
                ignore_index=-100,
            )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids:     torch.Tensor,   # (B, T_prompt) — the prompt tokens
        max_new_tokens: int = 80,
        eos_token_id:   int = 2,       # [EOS] token ID
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding.

        At each step: feed the current sequence into the model, take the
        token with the highest logit at the last position, append it to
        the sequence, and repeat until [EOS] or max_new_tokens is reached.

        Parameters
        ----------
        input_ids      : (B, T_prompt) — already tokenized prompt (up to [SEP])
        max_new_tokens : maximum number of new tokens to generate
        eos_token_id   : generation stops when this token is produced

        Returns
        -------
        (B, T_prompt + n_generated) — full sequence including prompt
        """
        self.eval()

        for _ in range(max_new_tokens):
            # If the current sequence is longer than our context window,
            # truncate from the LEFT to keep the most recent tokens.
            if input_ids.size(1) > self.config.n_positions:
                input_ids = input_ids[:, -self.config.n_positions:]

            _, logits = self.forward(input_ids)

            # Take the logits at the LAST position only (next-token prediction)
            next_logits = logits[:, -1, :]                        # (B, vocab_size)
            next_token  = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop early if all sequences in the batch have produced [EOS]
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def num_parameters(self) -> int:
        """
        Count trainable parameters.
        Subtracts lm_head.weight because it is shared with wte (weight tying)
        and would be double-counted by a naive sum.
        """
        n = sum(p.numel() for p in self.parameters())
        n -= self.wte.weight.numel()   # remove the shared copy
        return n

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GPT2BPE("
            f"n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
            f"n_embd={cfg.n_embd}, vocab={cfg.vocab_size}, "
            f"params={self.num_parameters()/1e6:.1f}M)"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run to verify the BPE model builds correctly and produces the right shapes:
        python model/gpt2_bpe.py
    """
    import yaml

    print("Building GPT2BPE model ...")

    config = GPT2Config(
        vocab_size  = 16000,
        n_positions = 256,
        n_layer     = 12,
        n_head      = 12,
        n_embd      = 768,
        dropout     = 0.1,
    )

    model = GPT2BPE(config)
    print(model)
    print(f"Parameters: {model.num_parameters()/1e6:.2f}M")

    # ── Forward pass test ─────────────────────────────────────────────────────
    B, T = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels    = input_ids.clone()
    # Mask first 50 positions (simulate input portion)
    labels[:, :50] = -100

    loss, logits = model(input_ids, labels=labels)

    print(f"\nForward pass:")
    print(f"  input_ids shape : {input_ids.shape}")
    print(f"  logits shape    : {logits.shape}")    # (2, 128, 16000)
    print(f"  loss            : {loss.item():.4f}")

    # ── Generation test ───────────────────────────────────────────────────────
    prompt = torch.randint(0, config.vocab_size, (1, 20))   # (1, 20)
    generated = model.generate(prompt, max_new_tokens=10, eos_token_id=2)
    print(f"\nGeneration test:")
    print(f"  prompt length    : {prompt.size(1)}")
    print(f"  generated length : {generated.size(1)}")

    # ── Weight tying check ────────────────────────────────────────────────────
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), \
        "Weight tying broken!"
    print(f"\nWeight tying: ✓  (lm_head and wte share the same weight tensor)")

    print("\n✓ GPT2BPE OK")
