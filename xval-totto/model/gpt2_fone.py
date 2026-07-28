"""
model/gpt2_fone.py
==================
GPT-2 with Fourier Number Embeddings (FoNE).

Paper: "FoNE: Precise Single-Token Number Embeddings via Fourier Features"
       (ICLR 2026), KevinZhoutianyi et al.

Key difference from BPE
-----------------------
At positions containing a number (token == [NUM]), the FoNE embedding of the
number's value is ADDED to the regular token embedding:

    x[t] = wte(token_id[t]) + wpe(t)                      for normal tokens
    x[t] = wte([NUM]) + FoNE(number_value[t]) + wpe(t)    for [NUM] tokens

The FoNE vector is 20-dimensional (int_digit_len=7, frac_digit_len=2 by default)
and is zero-padded to n_embd=768 before addition.

Key difference from xVal
-------------------------
- xVal MULTIPLIES the [NUM] embedding by the value (scaling)
- FoNE ADDS a separate Fourier-encoded vector to the [NUM] embedding (additive)
- FoNE has NO separate MSE regression loss — only standard CE loss
- Number value at inference time is decoded from the hidden state via FoNE
  inverse transform (atan2-based decoding), not a separate scalar head

Architecture:
  Input tokens (with [NUM] for numbers)
       ↓
  Token Embedding (wte) + Position Embedding (wpe)
       ↓
  *** FoNE: ADD Fourier embedding at [NUM] positions ***
       ↓
  GPT2Backbone (12 transformer layers — IDENTICAL to BPE model)
       ↓
  LM Head:      Linear(768, vocab_size)  [CE loss]
  FoNE Decoder: hidden vs target  [FoNE digit-wise CE loss at [NUM] positions]

Usage
-----
  from model.gpt2 import GPT2Config
  from model.gpt2_fone import GPT2FoNE

  config = GPT2Config()
  model  = GPT2FoNE(config, num_token_id=8, int_digit_len=7, frac_digit_len=2)
  loss, logits = model(input_ids, number_values, number_mask, labels=labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt2 import GPT2Config, GPT2Backbone, init_weights
from model.fone_encoder import FoNEEncoder


class GPT2FoNE(nn.Module):
    """
    GPT-2 with Fourier Number Embeddings (the FoNE model).

    Parameters
    ----------
    config         : GPT2Config — architecture hyperparameters (same as BPE)
    num_token_id   : int        — token ID of [NUM] in vocabulary (default 8)
    int_digit_len  : int        — max integer digits in numbers (default 7)
    frac_digit_len : int        — max fractional digits in numbers (default 2)
    add_linear     : bool       — add linear projection inside FoNEEncoder (default True)
    num_loss_weight: float      — weight for FoNE regression loss (default 1.0)
    """

    def __init__(
        self,
        config:         GPT2Config,
        num_token_id:   int   = 8,
        int_digit_len:  int   = 7,
        frac_digit_len: int   = 2,
        add_linear:     bool  = True,
        num_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.config          = config
        self.num_token_id    = num_token_id
        self.num_loss_weight = num_loss_weight

        # ── Embedding layers (same as BPE) ────────────────────────────────────
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # ── FoNE encoder (NEW — not in BPE model) ────────────────────────────
        # This maps raw float values → n_embd-dimensional Fourier embeddings.
        # Its output is added to the token embedding at [NUM] positions.
        self.fone = FoNEEncoder(
            n_embd         = config.n_embd,
            int_digit_len  = int_digit_len,
            frac_digit_len = frac_digit_len,
            add_linear     = add_linear,
        )

        # ── Shared transformer backbone (IDENTICAL to BPE) ────────────────────
        self.backbone = GPT2Backbone(config)

        # ── Language model head (same as BPE, weight-tied) ───────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying

        # ── Weight initialization ─────────────────────────────────────────────
        # Apply to all modules EXCEPT the FoNE encoder's linear (it has its own init)
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
        target_values: torch.Tensor = None,    # (B, T) float32 — target numbers at [NUM] label positions
    ) -> tuple:
        """
        Forward pass with FoNE number embedding.

        Parameters
        ----------
        input_ids     : (B, T) — token IDs; [NUM] tokens at number positions
        number_values : (B, T) — raw number values at [NUM] positions (0.0 elsewhere)
        number_mask   : (B, T) — True wherever input_ids == [NUM]
        labels        : (B, T) — -100 for input and padding; target token IDs elsewhere
        target_values : (B, T) — target numeric values at [NUM] label positions (0.0 elsewhere).
                        Used for the FoNE regression loss. If None, regression loss is skipped.

        Returns
        -------
        If labels is not None: (total_loss, logits)
          total_loss : CE loss + num_loss_weight * FoNE digit-wise CE loss
          logits     : (B, T, vocab_size)
        If labels is None: (None, logits)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Step 1: Standard BPE embedding lookup ─────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        tok_emb   = self.wte(input_ids)                           # (B, T, n_embd)
        pos_emb   = self.wpe(positions)                           # (1, T, n_embd)

        # ── Step 2: FoNE addition at [NUM] positions ──────────────────────────
        fone_emb = self.fone.encode_sequence(number_values, number_mask)  # (B, T, n_embd)
        x = tok_emb + pos_emb + fone_emb   # (B, T, n_embd)

        # ── Step 3: Transformer ───────────────────────────────────────────────
        hidden = self.backbone(x)   # (B, T, n_embd)

        # ── Step 4: LM head ───────────────────────────────────────────────────
        logits = self.lm_head(hidden)   # (B, T, vocab_size)

        # ── Step 5: Loss ──────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # 5a. CE loss over token sequence (shifted)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 5b. FoNE numeric digit-wise CE loss.
            #
            # At positions in the TARGET sequence where the label is [NUM],
            # the hidden state should encode the Fourier features of the target number.
            # We compute a digit-wise cross-entropy loss against the ground truth digits.
            #
            # This gives the model a gradient signal to encode the right numeric
            # value — without this, CE → 0 immediately but generated numbers
            # are arbitrary (the hidden state has no training pressure to encode
            # any particular value in FoNE space).
            #
            # We use the SHIFTED labels (labels[:, 1:]) to align with the shifted
            # logits, so the label at shifted position t corresponds to hidden[t].
            num_loss = torch.tensor(0.0, device=device)
            if target_values is not None and self.num_loss_weight > 0:
                # Identify positions where the target token is [NUM]
                # Use shifted labels to align with hidden states at prediction positions
                num_label_mask = (shift_labels == self.num_token_id)  # (B, T-1)

                if num_label_mask.any():
                    # Hidden states at [NUM] label positions (from teacher-forced input)
                    hidden_shifted = hidden[:, :-1, :]                    # (B, T-1, n_embd)
                    num_hidden     = hidden_shifted[num_label_mask]        # (N, n_embd)

                    # Target FoNE embeddings for those positions
                    # target_values is (B, T); shift to align with labels[:, 1:]
                    tgt_vals_shifted = target_values[:, 1:]               # (B, T-1)
                    tgt_vals_at_num  = tgt_vals_shifted[num_label_mask]   # (N,)

                    # Compute FoNE digit-wise CE loss
                    num_loss = self.fone.compute_loss(num_hidden, tgt_vals_at_num)

            loss = ce_loss + self.num_loss_weight * num_loss

        return loss, logits

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
        Autoregressive decoding with FoNE-based number prediction.

        For [NUM] tokens generated in the output:
          - The model's LM head decides WHEN to generate [NUM] (via CE training)
          - The hidden state at that position is decoded via FoNE inverse to
            recover the predicted number value

        Parameters
        ----------
        input_ids     : (B, T_prompt) — tokenized prompt
        number_values : (B, T_prompt) — raw values for [NUM] tokens in prompt
        number_mask   : (B, T_prompt) — True at [NUM] positions in prompt
        max_new_tokens: max tokens to generate
        eos_token_id  : generation stops when this token is produced
        repetition_penalty: penalty > 1.0 penalizes already seen tokens
        temperature   : sampling temperature (0.0 = greedy)

        Returns
        -------
        generated_ids    : (B, T_prompt + n_generated)
        predicted_values : list of length B, each a list of floats for
                           each [NUM] token generated in the output portion
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

            # Forward pass
            positions = torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
            tok_emb   = self.wte(idx)
            pos_emb   = self.wpe(positions)
            fone_emb  = self.fone.encode_sequence(nv, nm)
            x         = tok_emb + pos_emb + fone_emb
            hidden    = self.backbone(x)
            logits    = self.lm_head(hidden)

            # Next token prediction at last position
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    unique_tokens = input_ids[b].unique()
                    score = next_logits[b, unique_tokens]
                    score = torch.where(score < 0,
                                        score * repetition_penalty,
                                        score / repetition_penalty)
                    next_logits[b, unique_tokens] = score

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # For [NUM] tokens: decode number value from FoNE hidden state
            last_hidden = hidden[:, -1, :]   # (B, n_embd)
            for b in range(B):
                if next_token[b, 0].item() == self.num_token_id:
                    pred_val = self.fone.decode(last_hidden[b])
                    predicted_values[b].append(pred_val)

            # Append next token; new position has value=0 and mask=False
            input_ids     = torch.cat([input_ids, next_token], dim=1)
            new_nv        = torch.zeros(B, 1, device=input_ids.device)
            new_nm        = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
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
        return (
            f"GPT2FoNE("
            f"n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
            f"n_embd={cfg.n_embd}, vocab={cfg.vocab_size}, "
            f"fone_dim={self.fone.fone_dim}, "
            f"params={self.num_parameters()/1e6:.1f}M)"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building GPT2FoNE model ...")

    config = GPT2Config(
        vocab_size  = 16000,
        n_positions = 256,
        n_layer     = 12,
        n_head      = 12,
        n_embd      = 768,
        dropout     = 0.1,
    )

    model = GPT2FoNE(config, num_token_id=8, int_digit_len=7, frac_digit_len=2)
    print(model)
    print(f"Parameters: {model.num_parameters()/1e6:.2f}M")
    print(f"FoNE dim  : {model.fone.fone_dim}")

    # ── Forward pass test ─────────────────────────────────────────────────────
    B, T  = 2, 128
    NUM_ID = 8

    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels    = input_ids.clone()
    labels[:, :50] = -100

    # Simulate [NUM] tokens in the sequence
    number_mask = torch.zeros(B, T, dtype=torch.bool)
    number_mask[:, 10] = True
    number_mask[:, 60] = True
    number_mask[:, 80] = True
    input_ids[number_mask] = NUM_ID

    number_values = torch.zeros(B, T)
    number_values[number_mask] = torch.tensor([1874.0, 76.0, 42.0,
                                               2016.0, 99.0, 3.14])

    loss, logits = model(input_ids, number_values, number_mask, labels=labels)

    print(f"\nForward pass:")
    print(f"  input_ids shape : {input_ids.shape}")
    print(f"  logits shape    : {logits.shape}")
    print(f"  loss            : {loss.item():.4f}")

    # ── Weight tying check ────────────────────────────────────────────────────
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    print(f"\nWeight tying: ✓")

    # ── Generation test ───────────────────────────────────────────────────────
    prompt = torch.randint(0, config.vocab_size, (1, 20))
    pv     = torch.zeros(1, 20)
    pm     = torch.zeros(1, 20, dtype=torch.bool)
    gen_ids, gen_vals = model.generate(prompt, pv, pm, max_new_tokens=10)
    print(f"\nGeneration test:")
    print(f"  prompt length    : {prompt.size(1)}")
    print(f"  generated length : {gen_ids.size(1)}")
    print(f"  predicted [NUM] values: {gen_vals}")

    print("\n✓ GPT2FoNE OK")
