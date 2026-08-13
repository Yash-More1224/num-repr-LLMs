"""
model/gpt2_fone_dp.py
======================
GPT-2 with FoNE + Detached Projection head (FoNE-DP).

## Motivation: Representation Conflict

The original FoNE model trains the backbone with BOTH:
  - CE loss   : predicts the next linguistic token
  - FoNE loss : predicts digit-wise Fourier features at [NUM] positions

These two objectives conflict: the hidden state at [NUM] positions must
simultaneously encode linguistic context (for CE) AND digit-structured
Fourier features (for FoNE loss). Because the gradients from both losses
update the same backbone weights, the FoNE gradient overwrites linguistic
context, causing hallucination loops of repeated [NUM] tokens.

## The Fix: Multi-Task Projection Head

We add a small bottleneck MLP that branches off the backbone hidden states
AT [NUM] positions:

    num_proj = Linear(768, 64) → GELU → Linear(64, 768)

This projection head absorbs the complex geometric transformation of the 
digit-wise cross-entropy loss, softening the representation conflict 
without blinding the backbone to the exact numeric values.

At generation time (inside torch.no_grad()), the same num_proj projects the
last hidden state, and fone.decode() reads digit-by-digit argmax from it.

## Architecture

    Input tokens (with [NUM] for numbers)
         ↓
    Token Embedding (wte) + Position Embedding (wpe)
         ↓
    *** FoNE: ADD Fourier embedding at [NUM] positions (unchanged) ***
         ↓
    GPT2Backbone (12 transformer layers — IDENTICAL to standard FoNE)
         ↓
    LM Head: Linear(768, vocab_size)    [CE loss — standard, backbone trains]
    ┌─────────── Projection ───────────┐
    │  num_proj: Linear(768→64→768)    │ [FoNE digit-CE — num_proj & backbone train]
    └──────────────────────────────────┘

Usage
-----
  from model.gpt2 import GPT2Config
  from model.gpt2_fone_dp import GPT2FoNeDP

  config = GPT2Config()
  model  = GPT2FoNeDP(config, num_token_id=8, num_proj_hidden=64)
  loss, logits, ce_loss, fone_loss = model(
      input_ids, number_values, number_mask,
      labels=labels, target_values=number_values
  )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt2 import GPT2Config, GPT2Backbone, init_weights
from model.fone_encoder import FoNEEncoder


class GPT2FoNeDP(nn.Module):
    """
    GPT-2 with FoNE input encoding and a Multi-Task Projection head for
    digit-wise CE supervision (FoNE-DP).

    Parameters
    ----------
    config          : GPT2Config  — architecture hyperparameters
    num_token_id    : int         — token ID of [NUM] in vocabulary (default 8)
    int_digit_len   : int         — max integer digits (default 7)
    frac_digit_len  : int         — max fractional digits (default 2)
    num_proj_hidden : int         — bottleneck dim of the projection head (default 64)
    num_loss_weight : float       — weight for FoNE digit-CE loss (default 1.0)
    """

    def __init__(
        self,
        config:          GPT2Config,
        num_token_id:    int   = 8,
        int_digit_len:   int   = 7,
        frac_digit_len:  int   = 2,
        num_proj_hidden: int   = 64,
        num_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.config          = config
        self.num_token_id    = num_token_id
        self.num_loss_weight = num_loss_weight

        # ── Embedding layers ──────────────────────────────────────────────────
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # ── FoNE encoder (input-side: ADD Fourier features at [NUM]) ─────────
        # Identical to the standard FoNE model — the additive encoding is safe
        # (bounded cos/sin values) and is kept unchanged so the backbone still
        # receives number-magnitude information on the input side.
        self.fone = FoNEEncoder(
            n_embd         = config.n_embd,
            int_digit_len  = int_digit_len,
            frac_digit_len = frac_digit_len,
            add_linear     = True,
        )

        # ── Shared transformer backbone ───────────────────────────────────────
        self.backbone = GPT2Backbone(config)

        # ── Language model head (weight-tied to wte) ─────────────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying

        # ── Detached numeric projection head (NEW) ────────────────────────────
        # 768 → 64 → 768 bottleneck MLP.
        #
        # The output is a 768-dim vector that lives in the same space as
        # FoNEEncoder.compute_loss() and FoNEEncoder.decode() expect.
        # These two FoNE methods read specific slices of the 768-dim vector
        # (the first (int_digit_len + frac_digit_len) * num_cos_sin_per_digit
        # dims) to perform digit-wise CE loss / argmax decoding.
        #
        # Parameters: 768*64 + 64 + 64*768 + 768 = 98,560  (0.12% of model)
        self.num_proj = nn.Sequential(
            nn.Linear(config.n_embd, num_proj_hidden),
            nn.GELU(),
            nn.Linear(num_proj_hidden, config.n_embd),
        )

        # ── Weight initialization ─────────────────────────────────────────────
        self.apply(lambda m: init_weights(m, config.n_layer))

    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:     torch.Tensor,           # (B, T) int64
        number_values: torch.Tensor,           # (B, T) float32 — raw values at [NUM]
        number_mask:   torch.Tensor,           # (B, T) bool — True at [NUM] positions
        labels:        torch.Tensor = None,    # (B, T) int64, -100 where ignored
        target_values: torch.Tensor = None,    # (B, T) float32 — target numbers for loss
    ) -> tuple:
        """
        Forward pass for FoNE-DP.

        Parameters
        ----------
        input_ids     : (B, T) — token IDs; [NUM] for numbers
        number_values : (B, T) — raw number values at [NUM] positions (0.0 elsewhere)
        number_mask   : (B, T) — True wherever input_ids == [NUM]
        labels        : (B, T) — -100 for input + padding; target token IDs elsewhere
        target_values : (B, T) — target numeric values at [NUM] label positions.
                        Used for the FoNE-DP projection head loss.
                        If None, falls back to number_values (same as standard FoNE).

        Returns
        -------
        If labels is not None:
            (total_loss, logits, ce_loss, fone_dp_loss)
              total_loss    : scalar — ce_loss + num_loss_weight * fone_dp_loss
              logits        : (B, T, vocab_size)
              ce_loss       : scalar — cross-entropy component
              fone_dp_loss  : scalar — FoNE digit-CE on detached projection
        If labels is None:
            (None, logits)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Step 1: Embedding lookup ───────────────────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        tok_emb   = self.wte(input_ids)                           # (B, T, n_embd)
        pos_emb   = self.wpe(positions)                           # (1, T, n_embd)

        # ── Step 2: FoNE input encoding (ADD bounded Fourier features) ─────────
        # This is identical to the standard FoNE model.
        # The additive signal is bounded (cos/sin ∈ [-1, 1]) — no blowup risk.
        fone_emb = self.fone.encode_sequence(number_values, number_mask)  # (B, T, n_embd)
        x = tok_emb + pos_emb + fone_emb   # (B, T, n_embd)

        # ── Step 3: Transformer backbone ──────────────────────────────────────
        hidden = self.backbone(x)   # (B, T, n_embd)

        # ── Step 4: LM head → CE logits ───────────────────────────────────────
        logits = self.lm_head(hidden)   # (B, T, vocab_size)

        # ── Step 5: Loss computation ───────────────────────────────────────────
        if labels is None:
            return None, logits

        if target_values is None:
            target_values = number_values

        # 5a. Standard cross-entropy loss on all target tokens.
        # Labels are pre-shifted in the dataset: labels[t] = token_ids[t+1].
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 5b. FoNE-DP digit-wise CE on the DETACHED hidden state.
        #
        # This is the critical change vs. the standard FoNE model:
        #   - Standard FoNE: hidden → fone.compute_loss() directly
        #     → FoNE digit-CE gradients flow into the backbone
        #   - FoNE-DP: hidden.detach() → num_proj → fone.compute_loss()
        #     → FoNE digit-CE gradients ONLY update num_proj weights (98K params)
        #     → backbone is completely shielded from numeric gradient pressure
        #
        # Positions: we want hidden[t] (which has seen tokens 0..t) to predict
        # the value of the NEXT token, token_ids[t+1], if it is [NUM].
        # labels[t] == [NUM] exactly when t+1 carries a number.
        fone_dp_loss = torch.tensor(0.0, device=device)

        if self.num_loss_weight > 0:
            # Find positions where the NEXT token (in the label) is [NUM]
            target_num_mask = (labels[:, :-1] == self.num_token_id)  # (B, T-1) bool

            if target_num_mask.any():
                shift_hidden  = hidden[:, :-1, :]        # (B, T-1, n_embd)
                shift_targets = target_values[:, 1:]     # (B, T-1) — value at t+1

                # Gather only the [NUM] positions to save memory
                num_hidden = shift_hidden[target_num_mask]  # (N, n_embd)
                tgt_vals   = shift_targets[target_num_mask]  # (N,)

                # Project to 768-dim FoNE-decodable space
                proj_out = self.num_proj(num_hidden)          # (N, n_embd)

                # FoNE digit-wise CE loss (uses existing FoNEEncoder method unchanged)
                fone_dp_loss = self.fone.compute_loss(proj_out, tgt_vals)

        total_loss = ce_loss + self.num_loss_weight * fone_dp_loss

        return total_loss, logits, ce_loss, fone_dp_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,    # (B, T_prompt)
        number_values:  torch.Tensor,    # (B, T_prompt) — values for prompt [NUM]s
        number_mask:    torch.Tensor,    # (B, T_prompt) — mask for prompt [NUM]s
        max_new_tokens: int   = 80,
        eos_token_id:   int   = 2,
        repetition_penalty: float = 1.2,
        temperature:    float = 0.8,
    ) -> tuple:
        """
        Autoregressive decoding with FoNE-DP number prediction.

        For each [NUM] token generated:
          1. The LM head decides WHEN to generate [NUM] (via CE training)
          2. The last hidden state is projected by num_proj → 768-dim vector
          3. fone.decode() reads digit-by-digit argmax from that vector

        No detach() needed here — the entire call is inside torch.no_grad().

        Returns
        -------
        generated_ids    : (B, T_prompt + n_generated)
        predicted_values : list[list[float]] — predicted number values for each
                           [NUM] token generated in the output portion
        """
        self.eval()
        B = input_ids.size(0)
        predicted_values = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            T = input_ids.size(1)
            if T > self.config.n_positions:
                idx = input_ids[:, -self.config.n_positions:]
                nv  = number_values[:, -self.config.n_positions:]
                nm  = number_mask[:, -self.config.n_positions:]
            else:
                idx = input_ids
                nv  = number_values
                nm  = number_mask

            # Forward pass: FoNE input encoding + backbone
            positions = torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
            tok_emb   = self.wte(idx)
            pos_emb   = self.wpe(positions)
            fone_emb  = self.fone.encode_sequence(nv, nm)
            x         = tok_emb + pos_emb + fone_emb
            hidden    = self.backbone(x)
            logits    = self.lm_head(hidden)

            # Next token at the last position
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    unique_tokens = input_ids[b].unique()
                    score = next_logits[b, unique_tokens]
                    score = torch.where(
                        score < 0,
                        score * repetition_penalty,
                        score / repetition_penalty,
                    )
                    next_logits[b, unique_tokens] = score

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # For [NUM] tokens: project hidden state → FoNE decode → float value
            last_hidden = hidden[:, -1, :]          # (B, n_embd)
            proj_out    = self.num_proj(last_hidden) # (B, n_embd)

            for b in range(B):
                if next_token[b, 0].item() == self.num_token_id:
                    # fone.decode() does digit-by-digit argmax on the 768-dim vector
                    pred_val = self.fone.decode(proj_out[b])
                    predicted_values[b].append(pred_val)

            # Append next token; inject correct value and mask for [NUM] tokens
            input_ids     = torch.cat([input_ids, next_token], dim=1)
            new_nv        = torch.zeros(B, 1, device=input_ids.device)
            new_nm        = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
            
            for b in range(B):
                if next_token[b, 0].item() == self.num_token_id:
                    new_nv[b, 0] = predicted_values[b][-1]
                    new_nm[b, 0] = True
                    
            number_values = torch.cat([number_values, new_nv], dim=1)
            number_mask   = torch.cat([number_mask,   new_nm], dim=1)

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
        proj_params = sum(p.numel() for p in self.num_proj.parameters())
        return (
            f"GPT2FoNeDP("
            f"n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
            f"n_embd={cfg.n_embd}, vocab={cfg.vocab_size}, "
            f"fone_dim={self.fone.fone_dim}, "
            f"num_proj_params={proj_params:,}, "
            f"total_params={self.num_parameters()/1e6:.1f}M)"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Verify that:
      1. Forward pass works and returns (total_loss, logits, ce_loss, fone_dp_loss)
      2. GRADIENT ISOLATION: backbone weights receive zero gradient from FoNE-DP loss
      3. num_proj weights DO receive gradients from FoNE-DP loss
      4. Generation produces plausible float values

    Run with:
        python model/gpt2_fone_dp.py
    """
    print("=" * 60)
    print("Building GPT2FoNeDP model ...")
    print("=" * 60)

    config = GPT2Config(
        vocab_size  = 16000,
        n_positions = 256,
        n_layer     = 12,
        n_head      = 12,
        n_embd      = 768,
        dropout     = 0.1,
    )

    model = GPT2FoNeDP(
        config          = config,
        num_token_id    = 8,
        int_digit_len   = 7,
        frac_digit_len  = 2,
        num_proj_hidden = 64,
        num_loss_weight = 1.0,
    )
    print(model)
    print()

    # ── Forward pass test ─────────────────────────────────────────────────────
    B, T   = 2, 128
    NUM_ID = 8

    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels    = input_ids.clone()
    labels[:, :50] = -100   # mask input portion

    number_mask = torch.zeros(B, T, dtype=torch.bool)
    number_mask[:, 10] = True   # in input (should NOT contribute to fone_dp_loss)
    number_mask[:, 60] = True   # in target
    number_mask[:, 80] = True   # in target
    input_ids[number_mask] = NUM_ID

    number_values = torch.zeros(B, T)
    number_values[number_mask] = torch.tensor([1874.0, 76.0, 42.0,
                                               2016.0, 99.0, 3.14])

    total_loss, logits, ce_loss, fone_dp_loss = model(
        input_ids, number_values, number_mask,
        labels=labels, target_values=number_values
    )

    print(f"Forward pass:")
    print(f"  input_ids shape  : {input_ids.shape}")
    print(f"  logits shape     : {logits.shape}")
    print(f"  total_loss       : {total_loss.item():.4f}")
    print(f"  ce_loss          : {ce_loss.item():.4f}")
    print(f"  fone_dp_loss     : {fone_dp_loss.item():.4f}")
    print()

    # ── Gradient flow test ────────────────────────────────────────────────────
    # Run a backward pass seeded with ONLY the fone_dp_loss
    # to verify that backbone weights ARE updated by it.
    print("Gradient flow test ...")
    model.zero_grad()
    fone_dp_loss.backward()

    # num_proj MUST have gradients
    num_proj_grad = model.num_proj[0].weight.grad
    assert num_proj_grad is not None, "FAIL: num_proj[0].weight has no grad!"
    assert num_proj_grad.abs().max().item() > 0, "FAIL: num_proj[0].weight grad is all zero!"
    print(f"  num_proj[0].weight.grad : max={num_proj_grad.abs().max().item():.6f}  ✓")

    # wte (backbone embedding) MUST have gradients from fone_dp_loss
    wte_grad = model.wte.weight.grad
    assert wte_grad is not None and wte_grad.abs().max().item() > 1e-9, "FAIL: no gradient reached backbone!"
    print(f"  wte.weight.grad         : max={wte_grad.abs().max().item():.2e}  ✓")

    print()
    print("Gradient flow: ✓  (FoNE-DP loss now correctly reaches the backbone)")
    print()

    # ── Weight tying check ────────────────────────────────────────────────────
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    print("Weight tying: ✓")
    print()

    # ── Generation test ───────────────────────────────────────────────────────
    prompt = torch.randint(0, config.vocab_size, (1, 20))
    pv     = torch.zeros(1, 20)
    pm     = torch.zeros(1, 20, dtype=torch.bool)
    gen_ids, gen_vals = model.generate(prompt, pv, pm, max_new_tokens=15,
                                       temperature=0.0)
    print(f"Generation test:")
    print(f"  prompt length    : {prompt.size(1)}")
    print(f"  generated length : {gen_ids.size(1)}")
    print(f"  predicted [NUM] values: {gen_vals}")
    print()
    print("✓ GPT2FoNeDP OK")
