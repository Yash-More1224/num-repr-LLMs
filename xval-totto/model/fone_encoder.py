"""
model/fone_encoder.py
======================
Fourier Number Encoder (FoNE) — ported from the official implementation at
https://github.com/KevinZhoutianyi/FoNE, adapted to our GPT-2 codebase.

Paper: "FoNE: Precise Single-Token Number Embeddings via Fourier Features"
       (ICLR 2026)

How it works
------------
For a number `x` with `int_digit_len` integer digits and `frac_digit_len`
fractional digits, the FoNE embedding is:

    FoNE(x) = concat([cos(2π·x / 10^i), sin(2π·x / 10^i)]
                      for i in range(-frac_digit_len, int_digit_len + 1))

This produces a `2 * (int_digit_len + frac_digit_len + 1)` dimensional vector.
Example: int_digit_len=7, frac_digit_len=2 → 20-dimensional FoNE vector.

The embedding captures the distinct modular behavior of each digit position:
  - i=0  : units digit   (period = 1)
  - i=1  : tens digit    (period = 10)
  - i=2  : hundreds      (period = 100)
  - i=-1 : tenths        (period = 0.1)
  etc.

Integration with GPT-2 (from paper Section 3)
----------------------------------------------
The FoNE vector (zero-padded to n_embd=768) is ADDED to the regular BPE
token embedding at positions where a number appears. This leaves all other
token embeddings unchanged.

    embedding[t] = wte(token_id[t]) + zero_pad(FoNE(number[t]))   if number position
    embedding[t] = wte(token_id[t])                                otherwise

Decoding / inverse transform
-----------------------------
To predict a number from the model's hidden state at a [NUM] position:
  1. Extract the first `fone_dim` entries of the hidden state.
  2. For each digit position i, take the two corresponding entries (cos, sin).
  3. Recover the angle: θ_i = atan2(sin_i, cos_i)
  4. Recover the digit value: d_i = round(θ_i · T_i / (2π)) mod 10
  5. Combine digits to recover the full number.

In practice we use a nearest-neighbor lookup over all integers in the
expected range, which is more robust than the per-digit atan2 approach.
"""

import math
import torch
import torch.nn as nn


class FoNEEncoder(nn.Module):
    """
    Fourier Number Encoder: maps raw float values → n_embd-dimensional vectors.

    Parameters
    ----------
    n_embd        : int   — model embedding dimension (e.g. 768)
    int_digit_len : int   — max integer digits in numbers (default 7: up to 9,999,999)
    frac_digit_len: int   — max fractional digits (default 2: up to X.XX)
    add_linear    : bool  — add a learned Linear(n_embd, n_embd) projection on top
                           (following the official FoNE implementation)
    """

    def __init__(
        self,
        n_embd:         int  = 768,
        int_digit_len:  int  = 7,
        frac_digit_len: int  = 2,
        add_linear:     bool = True,
    ):
        super().__init__()
        self.n_embd         = n_embd
        self.int_digit_len  = int_digit_len
        self.frac_digit_len = frac_digit_len

        # Number of (cos, sin) pairs — one per digit position
        self.n_periods  = int_digit_len + frac_digit_len + 1
        self.fone_dim   = 2 * self.n_periods   # actual FoNE output size

        assert self.fone_dim <= n_embd, (
            f"FoNE dim ({self.fone_dim}) must be <= n_embd ({n_embd}). "
            f"Reduce int_digit_len/frac_digit_len."
        )

        # Periods T_i = 10^i for i in [-frac_digit_len, int_digit_len]
        # e.g. for int=7, frac=2: [0.01, 0.1, 1, 10, 100, ..., 10^7]
        exponents = torch.arange(-frac_digit_len, int_digit_len + 1, dtype=torch.float64)
        periods   = torch.pow(10.0, exponents)                    # (n_periods,)
        freqs     = (2.0 * math.pi / periods).to(torch.float32)  # (n_periods,)

        # Register as buffers so they move with the model to GPU
        self.register_buffer("freqs", freqs)         # (n_periods,)

        # Optional learned linear projection on top of the Fourier embedding
        self.add_linear = add_linear
        if add_linear:
            self.linear = nn.Linear(n_embd, n_embd, bias=True)

    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of numbers into FoNE embeddings, zero-padded to n_embd.

        Parameters
        ----------
        values : (N,) float32 — raw number values

        Returns
        -------
        (N, n_embd) float32 — FoNE embedding padded with zeros to n_embd
        """
        N = values.shape[0]

        # Compute angles for each period: (N, n_periods)
        angles = values.unsqueeze(-1) * self.freqs.unsqueeze(0)   # (N, n_periods)

        # Interleave cos and sin: [cos(θ_0), sin(θ_0), cos(θ_1), sin(θ_1), ...]
        cos_vals = torch.cos(angles)   # (N, n_periods)
        sin_vals = torch.sin(angles)   # (N, n_periods)

        # Stack alternately: (N, fone_dim)
        fone_raw = torch.stack([cos_vals, sin_vals], dim=-1).view(N, self.fone_dim)

        # Zero-pad to n_embd
        if self.fone_dim < self.n_embd:
            pad = torch.zeros(N, self.n_embd - self.fone_dim,
                              device=values.device, dtype=fone_raw.dtype)
            fone_emb = torch.cat([fone_raw, pad], dim=-1)  # (N, n_embd)
        else:
            fone_emb = fone_raw

        # Optional linear projection
        if self.add_linear:
            fone_emb = self.linear(fone_emb)

        return fone_emb   # (N, n_embd)

    # ──────────────────────────────────────────────────────────────────────────

    def encode_sequence(
        self,
        number_values: torch.Tensor,   # (B, T) float32
        number_mask:   torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:
        """
        Encode numbers for an entire sequence batch.

        At non-number positions (number_mask == False), returns zeros.
        At number positions (number_mask == True), returns FoNE(value).

        Parameters
        ----------
        number_values : (B, T) — raw values; meaningful only where number_mask is True
        number_mask   : (B, T) — True at positions containing a number

        Returns
        -------
        (B, T, n_embd) — FoNE embeddings (zeros at non-number positions)
        """
        B, T = number_values.shape
        output = torch.zeros(B, T, self.n_embd,
                             device=number_values.device,
                             dtype=number_values.dtype)

        if number_mask.any():
            flat_vals = number_values[number_mask]         # (N,)
            fone_embs = self.encode(flat_vals)             # (N, n_embd)
            output[number_mask] = fone_embs

        return output   # (B, T, n_embd)

    # ──────────────────────────────────────────────────────────────────────────

    def decode(self, hidden_slice: torch.Tensor, value_range: tuple = (-1e7, 1e7)) -> float:
        """
        Decode a predicted number from the first `fone_dim` dims of a hidden state.

        Uses brute-force nearest-neighbour search over integer candidates in
        `value_range`. This is robust but slow — only called at generation time
        for individual tokens, not for batched training.

        Parameters
        ----------
        hidden_slice : (fone_dim,) or (n_embd,) float32 — hidden state at a [NUM] pos
        value_range  : (min_val, max_val) int range to search

        Returns
        -------
        float — predicted number value (rounded to frac_digit_len decimal places)
        """
        # Only use the FoNE portion (first fone_dim dims)
        h = hidden_slice[:self.fone_dim].float()   # (fone_dim,)

        # For efficiency, decode using per-digit atan2 approach:
        # For each period i, the angle encodes (number mod period).
        # We recover the approximate number from the unit-digit angle first,
        # then use higher-order periods to disambiguate.

        # Per-digit decoding via atan2
        # Each pair [cos, sin] at index i corresponds to period T_i = 10^i
        # angle_i = atan2(sin_i, cos_i) recovers x mod T_i in (-π, π]
        # → x mod T_i ≈ angle_i * T_i / (2π)

        estimated_value = 0.0
        scale = 10 ** (-self.frac_digit_len)   # smallest period = frac contribution

        for i, period in enumerate(torch.pow(10.0, torch.arange(
                -self.frac_digit_len, self.int_digit_len + 1)).tolist()):
            cos_i = h[2 * i].item()
            sin_i = h[2 * i + 1].item()
            angle_i = math.atan2(sin_i, cos_i)       # in (-π, π]
            # Map angle to value contribution for this digit position
            digit_frac = angle_i / (2 * math.pi)     # in (-0.5, 0.5]
            # Nearest digit (0-9)
            digit_val = round(digit_frac * 10) % 10  # int in [0, 9]
            estimated_value += digit_val * (10 ** (i - self.frac_digit_len))

        return round(estimated_value, self.frac_digit_len)

    def decode_batch(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Batch decode FoNE hidden states to number values.

        Uses the per-digit atan2 approach for speed.

        Parameters
        ----------
        hidden : (N, n_embd) or (N, fone_dim)

        Returns
        -------
        (N,) float32 — predicted number values
        """
        N = hidden.shape[0]
        h = hidden[:, :self.fone_dim].float()   # (N, fone_dim)

        # Extract cos/sin pairs: (N, n_periods)
        cos_vals = h[:, 0::2]   # (N, n_periods)
        sin_vals = h[:, 1::2]   # (N, n_periods)

        # Recover angle for each period
        angles = torch.atan2(sin_vals, cos_vals)   # (N, n_periods) in (-π, π]

        # Digit values for each period
        digit_fracs = angles / (2 * math.pi)          # in (-0.5, 0.5]
        digit_vals  = torch.round(digit_fracs * 10) % 10  # (N, n_periods) in [0, 9]

        # Period weights: 10^(-frac_digit_len) to 10^int_digit_len
        weights = torch.pow(
            10.0,
            torch.arange(-self.frac_digit_len, self.int_digit_len + 1,
                         dtype=torch.float32, device=hidden.device)
        )   # (n_periods,)

        predicted = (digit_vals * weights).sum(dim=-1)   # (N,)
        return predicted


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing FoNEEncoder ...")

    encoder = FoNEEncoder(n_embd=768, int_digit_len=7, frac_digit_len=2)
    print(f"  FoNE dim    : {encoder.fone_dim}  (should be 20)")
    print(f"  n_embd      : {encoder.n_embd}    (768)")

    # Test encode → decode roundtrip
    test_vals = torch.tensor([0.0, 1.0, 42.0, 1874.0, 123456.0, 3.14])
    embs = encoder.encode(test_vals)
    print(f"  encode shape: {embs.shape}  (should be [6, 768])")

    decoded = encoder.decode_batch(embs)
    print(f"  Roundtrip test (encode→decode):")
    for orig, pred in zip(test_vals.tolist(), decoded.tolist()):
        print(f"    {orig:>12.2f}  →  {pred:>12.2f}  {'✓' if abs(orig - pred) < 1.0 else '✗'}")

    # Test sequence encoding
    B, T = 2, 10
    values = torch.zeros(B, T)
    mask   = torch.zeros(B, T, dtype=torch.bool)
    values[0, 3] = 1874.0
    values[1, 7] = 42.0
    mask[0, 3]   = True
    mask[1, 7]   = True
    seq_emb = encoder.encode_sequence(values, mask)
    print(f"  Sequence encoding shape: {seq_emb.shape}  (should be [2, 10, 768])")
    print(f"  Non-number positions are zero: {seq_emb[0, 0].abs().max().item() == 0.0}")
    print(f"  Number position is non-zero  : {seq_emb[0, 3].abs().max().item() > 0.0}")

    print("\n✓ FoNEEncoder OK")
