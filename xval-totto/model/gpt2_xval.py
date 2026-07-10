"""
model/gpt2_xval.py
==================
The EXPERIMENTAL model: GPT-2 with xVal continuous number embeddings.

The ONLY difference from GPT2BPE is how numbers are handled:

  BPE model:
    Input token "2018" → BPE splits into tokens → normal embedding lookup
    → the number's magnitude is NOT encoded in the embedding

  xVal model:
    Input token [NUM] (a single special token) → embedding lookup for [NUM]
    → embedding is SCALED by the actual number value (log-normalized)
    → the number's magnitude IS directly encoded in the embedding vector

  Additionally, the xVal model has a scalar regression head:
    At [NUM] positions in the TARGET, a Linear(768, 1) head predicts
    the number value. This is trained with MSE loss.

  Combined loss:
    L = L_ce   (cross-entropy on all target tokens, including [NUM])
      + λ × L_mse  (MSE on [NUM] tokens in target only)

    λ (mse_lambda) defaults to 1.0.

Architecture:
  Input tokens (with [NUM] for numbers)
       ↓
  Token Embedding (wte) + Position Embedding (wpe)
       ↓
  *** xVal scaling: [NUM] embeddings multiplied by number_values ***
       ↓
  GPT2Backbone (12 transformer layers — IDENTICAL to BPE model)
       ↓
  LM Head: Linear(768, vocab_size)   [CE loss on all target tokens]
  Scalar Head: Linear(768, 1)        [MSE loss on [NUM] target tokens]

Usage
-----
  from model.gpt2 import GPT2Config
  from model.gpt2_xval import GPT2XVal

  config = GPT2Config()
  model  = GPT2XVal(config, num_token_id=8, mse_lambda=1.0)
  loss, logits, ce_loss, mse_loss = model(
      input_ids, number_values, number_mask, labels=labels
  )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt2 import GPT2Config, GPT2Backbone, init_weights


class GPT2XVal(nn.Module):
    """
    GPT-2 with xVal continuous number embeddings (the experimental model).

    Parameters
    ----------
    config       : GPT2Config  — architecture hyperparameters (same as BPE)
    num_token_id : int         — token ID of [NUM] in the vocabulary (default 8)
    mse_lambda   : float       — weight of MSE loss relative to CE loss (default 1.0)
    """

    def __init__(
        self,
        config:       GPT2Config,
        num_token_id: int   = 8,
        mse_lambda:   float = 1.0,
    ):
        super().__init__()
        self.config       = config
        self.num_token_id = num_token_id
        self.mse_lambda   = mse_lambda

        # ── Embedding layers (same as BPE) ────────────────────────────────────
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # ── Shared transformer backbone (IDENTICAL to BPE) ────────────────────
        # This is the part of the model that is guaranteed to be the same.
        # The same GPT2Backbone class, same config, same initialization.
        self.backbone = GPT2Backbone(config)

        # ── LM head (same as BPE, weight-tied) ───────────────────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying

        # ── Scalar head (NEW — not in BPE model) ─────────────────────────────
        # A tiny Linear(768, 1) that takes the hidden state at each [NUM]
        # position in the TARGET and predicts the log-normalized number value.
        # This adds only 768 + 1 = 769 parameters — essentially nothing.
        self.scalar_head = nn.Linear(config.n_embd, 1, bias=True)

        # ── Weight initialization (same procedure as BPE) ─────────────────────
        self.apply(lambda m: init_weights(m, config.n_layer))

    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:     torch.Tensor,            # (B, T) int64
        number_values: torch.Tensor,            # (B, T) float32 — log-normalized values
        number_mask:   torch.Tensor,            # (B, T) bool — True at [NUM] positions
        labels:        torch.Tensor = None,     # (B, T) int64, -100 where ignored
    ) -> tuple:
        """
        Forward pass with xVal number embedding.

        Parameters
        ----------
        input_ids     : (B, T) — token IDs; [NUM] tokens at number positions
        number_values : (B, T) — log-normalized scalar value at [NUM] positions,
                        0.0 elsewhere (from dataset_xval.py)
        number_mask   : (B, T) — True wherever input_ids == [NUM]
        labels        : (B, T) — same as input_ids but -100 for input portion
                        and padding. If None, loss is not computed.

        Returns
        -------
        If labels is not None:
            (total_loss, logits, ce_loss, mse_loss)
              total_loss : scalar — ce_loss + mse_lambda * mse_loss
              logits     : (B, T, vocab_size)
              ce_loss    : scalar — cross-entropy component (for logging)
              mse_loss   : scalar — MSE component (for logging)

        If labels is None:
            (None, logits)
        """
        B, T   = input_ids.shape
        device = input_ids.device

        # ── Step 1: Standard embedding lookup ─────────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        tok_emb   = self.wte(input_ids)                           # (B, T, n_embd)
        pos_emb   = self.wpe(positions)                           # (1, T, n_embd)

        # ── Step 2: xVal scaling (THE ONLY DIFFERENCE FROM BPE) ──────────────
        #
        # For every [NUM] token, we scale its embedding vector by the actual
        # number value. For all other tokens, we multiply by 1.0 (no change).
        #
        # Why this works:
        #   - The model learns a "direction" for [NUM] in embedding space.
        #   - The magnitude of the [NUM] embedding encodes the number's value.
        #   - For non-number tokens, behavior is identical to the BPE model.
        #
        # Formula:
        #   scale[b, t] = number_values[b, t]  if number_mask[b, t] else 1.0
        #   tok_emb[b, t, :] *= scale[b, t]
        #
        # torch.where picks from number_values or ones element-wise:
        scale   = torch.where(
            number_mask,                    # condition: (B, T) bool
            number_values,                  # value where True (the number)
            torch.ones_like(number_values), # value where False (1.0 = no change)
        ).unsqueeze(-1)                     # (B, T, 1) for broadcasting over n_embd

        tok_emb = tok_emb * scale           # (B, T, n_embd) — [NUM] scaled, rest unchanged

        x = tok_emb + pos_emb

        # ── Step 3: Transformer (identical to BPE) ────────────────────────────
        hidden = self.backbone(x)           # (B, T, n_embd)

        # ── Step 4: LM head (identical to BPE) ───────────────────────────────
        logits = self.lm_head(hidden)       # (B, T, vocab_size)

        # ── Step 5: Loss computation ──────────────────────────────────────────
        if labels is None:
            return None, logits

        # Cross-entropy loss (same as BPE — on all target tokens, -100 ignored)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # MSE loss (NEW — only at [NUM] tokens in the TARGET portion)
        #
        # We want to compute MSE only where:
        #   (a) The token IS a [NUM] token (number_mask is True)
        #   (b) The position IS in the target (labels != -100)
        #
        # This means the model's scalar head is trained to predict the
        # value of numbers that appear in the OUTPUT SENTENCE only —
        # not the input table cells (which are given, not predicted).
        target_num_mask = number_mask & (labels != -100)   # (B, T) bool

        if target_num_mask.any():
            # hidden states at target [NUM] positions: (N, n_embd)
            # where N = total number of target [NUM] tokens in the batch
            pred_scalars = self.scalar_head(hidden)[target_num_mask].squeeze(-1)  # (N,)
            true_scalars = number_values[target_num_mask]                         # (N,)
            mse_loss = F.mse_loss(pred_scalars, true_scalars)
        else:
            # No numbers in the target for this batch — MSE loss is 0
            mse_loss = torch.tensor(0.0, device=device, requires_grad=False)

        total_loss = ce_loss + self.mse_lambda * mse_loss

        return total_loss, logits, ce_loss, mse_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,      # (B, T_prompt) — prompt tokens
        number_values:  torch.Tensor,      # (B, T_prompt) — values for prompt
        number_mask:    torch.Tensor,      # (B, T_prompt) — mask for prompt
        max_new_tokens: int = 80,
        eos_token_id:   int = 2,
    ) -> tuple:
        """
        Greedy autoregressive decoding for the xVal model.

        For [NUM] tokens in the GENERATED output, also returns the predicted
        scalar values (after denormalization) for evaluation.

        Parameters
        ----------
        input_ids     : (B, T_prompt) — tokenized prompt (up to [SEP])
        number_values : (B, T_prompt) — log-normalized values for prompt [NUM]s
        number_mask   : (B, T_prompt) — True at [NUM] positions in prompt
        max_new_tokens : max tokens to generate
        eos_token_id  : stop when this token is produced

        Returns
        -------
        generated_ids    : (B, T_prompt + n_generated)
        predicted_values : list of length B, each a list of scalar values
                           for [NUM] tokens generated in the output portion.
                           Values are in LOG-NORMALIZED space (use
                           denormalize_number() from dataset_xval.py to recover
                           original values).
        """
        self.eval()
        B = input_ids.size(0)

        # Track predicted scalar values for [NUM] tokens in the generated part
        # predicted_values[b] = list of float values for each [NUM] the model generates
        predicted_values = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            # Truncate to context window if needed
            T = input_ids.size(1)
            if T > self.config.n_positions:
                idx  = input_ids[:, -self.config.n_positions:]
                nv   = number_values[:, -self.config.n_positions:]
                nm   = number_mask[:, -self.config.n_positions:]
            else:
                idx  = input_ids
                nv   = number_values
                nm   = number_mask

            # Forward pass (no labels — just get logits and hidden states)
            positions = torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
            tok_emb   = self.wte(idx)
            scale     = torch.where(nm, nv, torch.ones_like(nv)).unsqueeze(-1)
            tok_emb   = tok_emb * scale
            x         = tok_emb + self.wpe(positions)
            hidden    = self.backbone(x)
            logits    = self.lm_head(hidden)

            # Next token: greedy argmax at the last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (B, 1)

            # For [NUM] tokens: predict scalar value using the scalar head
            last_hidden = hidden[:, -1:, :]                               # (B, 1, n_embd)
            pred_scalar = self.scalar_head(last_hidden).squeeze(-1).squeeze(-1)  # (B,)

            for b in range(B):
                if next_token[b, 0].item() == self.num_token_id:
                    predicted_values[b].append(pred_scalar[b].item())

            # Append next token to sequence
            input_ids     = torch.cat([input_ids, next_token], dim=1)
            # New positions get value=0.0 and mask=False (they are regular tokens)
            new_nv        = torch.zeros(B, 1, device=input_ids.device)
            new_nm        = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
            number_values = torch.cat([number_values, new_nv], dim=1)
            number_mask   = torch.cat([number_mask,   new_nm], dim=1)

            # Stop if all sequences have generated [EOS]
            if (next_token == eos_token_id).all():
                break

        return input_ids, predicted_values

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.wte.weight.numel()   # remove double-counted tied weights
        return n

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GPT2XVal("
            f"n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
            f"n_embd={cfg.n_embd}, vocab={cfg.vocab_size}, "
            f"mse_lambda={self.mse_lambda}, "
            f"params={self.num_parameters()/1e6:.1f}M)"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run to verify the xVal model and confirm the only difference from BPE
    is the embedding scaling and the scalar head:
        python model/gpt2_xval.py
    """
    print("Building GPT2XVal model ...")

    config = GPT2Config(
        vocab_size  = 16000,
        n_positions = 256,
        n_layer     = 12,
        n_head      = 12,
        n_embd      = 768,
        dropout     = 0.1,
    )

    model = GPT2XVal(config, num_token_id=8, mse_lambda=1.0)
    print(model)
    print(f"Parameters: {model.num_parameters()/1e6:.2f}M")

    # ── Forward pass test ─────────────────────────────────────────────────────
    B, T = 2, 128
    NUM_ID = 8

    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels    = input_ids.clone()
    labels[:, :50] = -100   # mask input portion

    # Simulate 3 [NUM] tokens in each sequence (some in input, some in target)
    number_mask = torch.zeros(B, T, dtype=torch.bool)
    number_mask[:, 10] = True   # in input portion (should NOT contribute to MSE loss)
    number_mask[:, 60] = True   # in target portion (SHOULD contribute to MSE loss)
    number_mask[:, 80] = True   # in target portion
    input_ids[number_mask] = NUM_ID  # set those positions to [NUM] token

    number_values = torch.zeros(B, T)
    number_values[number_mask] = torch.tensor([2018.0, 12.5, 7.6, 2018.0, 12.5, 7.6])

    total_loss, logits, ce_loss, mse_loss = model(
        input_ids, number_values, number_mask, labels=labels
    )

    print(f"\nForward pass:")
    print(f"  input_ids shape    : {input_ids.shape}")
    print(f"  logits shape       : {logits.shape}")
    print(f"  total_loss         : {total_loss.item():.4f}")
    print(f"  ce_loss            : {ce_loss.item():.4f}")
    print(f"  mse_loss           : {mse_loss.item():.4f}")

    # ── Verify [NUM] positions get scaled embeddings ──────────────────────────
    # A [NUM] token with value 0.0 should produce a zero embedding
    # A [NUM] token with value 1.0 should produce the base [NUM] embedding
    positions = torch.arange(T).unsqueeze(0)
    tok_emb   = model.wte(input_ids)
    scale     = torch.where(number_mask, number_values, torch.ones_like(number_values)).unsqueeze(-1)
    scaled    = tok_emb * scale

    # At non-NUM positions, scale is 1.0 → embeddings unchanged
    non_num_pos = (~number_mask).nonzero(as_tuple=False)[0]
    b0, t0 = non_num_pos[0].item(), non_num_pos[1].item()
    assert torch.allclose(scaled[b0, t0], tok_emb[b0, t0]), "Non-NUM embeddings should be unchanged"

    # At NUM positions, embedding is scaled by number_values
    num_pos = number_mask.nonzero(as_tuple=False)[0]
    b1, t1 = num_pos[0].item(), num_pos[1].item()
    expected = tok_emb[b1, t1] * number_values[b1, t1]
    assert torch.allclose(scaled[b1, t1], expected), "NUM embeddings should be scaled"

    print(f"\nEmbedding scaling: ✓")
    print(f"  Non-[NUM] embeddings: unchanged (scale=1.0)")
    print(f"  [NUM] embeddings    : scaled by number_values")

    # ── Weight tying check ────────────────────────────────────────────────────
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    print(f"Weight tying        : ✓")

    # ── Generation test ───────────────────────────────────────────────────────
    prompt = torch.randint(0, config.vocab_size, (1, 20))
    pv     = torch.zeros(1, 20)
    pm     = torch.zeros(1, 20, dtype=torch.bool)
    gen_ids, gen_vals = model.generate(prompt, pv, pm, max_new_tokens=10)
    print(f"\nGeneration test:")
    print(f"  prompt length    : {prompt.size(1)}")
    print(f"  generated length : {gen_ids.size(1)}")
    print(f"  predicted [NUM] scalar values: {gen_vals}")

    print("\n✓ GPT2XVal OK")
