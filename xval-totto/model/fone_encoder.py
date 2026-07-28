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
fractional digits, the FoNE embedding captures the modular arithmetic 
at each digit position by scaling periods by basis [2, 5].

Integration with GPT-2
----------------------
The FoNE vector (zero-padded to n_embd) is ADDED to the regular BPE
token embedding at positions where a number appears.

Decoding / inverse transform
-----------------------------
To predict a number from the model's hidden state at a [NUM] position,
the hidden state is multiplied by the precomputed Cos/Sin Fourier matrix
to generate a 10-way logit space per digit, which is then decoded via argmax.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FoNEEncoder(nn.Module):
    """
    Fourier Number Encoder: maps raw float values → n_embd-dimensional vectors.

    Parameters
    ----------
    n_embd        : int   — model embedding dimension (e.g. 768)
    int_digit_len : int   — max integer digits in numbers (default 7)
    frac_digit_len: int   — max fractional digits (default 2)
    add_linear    : bool  — add a learned Linear(n_embd, n_embd) projection
    """

    def __init__(
        self,
        n_embd:         int  = 768,
        int_digit_len:  int  = 7,
        frac_digit_len: int  = 2,
        add_linear:     bool = True,
        period_base_list: list = [2, 5],
    ):
        super().__init__()
        self.n_embd         = n_embd
        self.int_digit_len  = int_digit_len
        self.frac_digit_len = frac_digit_len
        self.add_linear     = add_linear

        # Initialize period list for Fourier embedding
        self.period_list = self._get_period_list(
            period_base_list, 
            minvalue=10**(-frac_digit_len), 
            maxvalue=10**(int_digit_len+1)-1
        )
        self.period_base_list_len = len(period_base_list)

        # Create and register the precomputed cosine/sine matrix as a buffer
        with torch.no_grad():
            precomputed_cos_sin_matrix = self._create_precomputed_cos_sin_matrix(period_base_list).T
        self.register_buffer("precomputed_cos_sin_matrix", precomputed_cos_sin_matrix)

        # Precompute the cosine/sine frequencies and register as a buffer
        period_tensor = torch.tensor(self.period_list, dtype=torch.float64)
        self.register_buffer("period_tensor", period_tensor)
        frequencies = (2 * math.pi / period_tensor)
        self.register_buffer("frequencies", frequencies)

        # Initialize other attributes
        if self.add_linear:
            self.linear = nn.Linear(self.n_embd, self.n_embd) 
        
        self.layer_norm = nn.LayerNorm(n_embd, eps=1e-5)

        # Precompute powers of ten for digit manipulation
        self.max_num_digits = int_digit_len + frac_digit_len
        powers_of_ten = torch.pow(10, torch.arange(self.max_num_digits)).long()[-(int_digit_len+frac_digit_len):]
        self.register_buffer("powers_of_ten", powers_of_ten)
        self.register_buffer("powers_of_ten_flipped", powers_of_ten.flip(0))

    @property
    def fone_dim(self) -> int:
        return 2 * len(self.period_list)

    def _get_period_list(self, period_base_list, minvalue, maxvalue):
        period_list = set()
        
        # Generate fractional periods
        current_value = 1.0
        while current_value >= minvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue:
                    period_list.add(period)
            current_value /= 10.0
            
        # Generate integer periods
        current_value = 1.0
        while current_value <= maxvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue:
                    period_list.add(period)
            current_value *= 10.0
            
        return sorted(list(period_list))

    def _create_precomputed_cos_sin_matrix(self, period_base_list):
        num_positions = 10
        base_positions = torch.arange(num_positions, dtype=torch.float32)
        cos_sin_list = []
        for period in period_base_list:
            w = 2 * math.pi / period
            cos_sin_list.append(torch.cos(w * base_positions))
            cos_sin_list.append(torch.sin(w * base_positions))
        return torch.stack(cos_sin_list, dim=1)

    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of numbers into FoNE embeddings."""
        N = values.shape[0]

        # Transform numbers into cos/sin based embeddings
        angles = values.unsqueeze(-1).to(torch.float64) * self.frequencies.unsqueeze(0)
        
        cos_vals = torch.cos(angles).to(torch.float32)
        sin_vals = torch.sin(angles).to(torch.float32)

        fone_raw = torch.stack([cos_vals, sin_vals], dim=-1).view(N, self.fone_dim)

        if self.fone_dim < self.n_embd:
            pad = torch.zeros(N, self.n_embd - self.fone_dim, device=values.device, dtype=fone_raw.dtype)
            fone_emb = torch.cat([fone_raw, pad], dim=-1)
        else:
            fone_emb = fone_raw

        if self.add_linear:
            fone_emb = self.linear(fone_emb)

        return fone_emb

    def encode_sequence(
        self,
        number_values: torch.Tensor,   # (B, T) float32
        number_mask:   torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:
        """Encode numbers for an entire sequence batch."""
        B, T = number_values.shape
        output = torch.zeros(B, T, self.n_embd,
                             device=number_values.device,
                             dtype=number_values.dtype)

        if number_mask.any():
            flat_vals = number_values[number_mask]
            fone_embs = self.encode(flat_vals)
            output[number_mask] = fone_embs.to(output.dtype)

        return output

    # ──────────────────────────────────────────────────────────────────────────

    def compute_loss(self, hidden_state: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute FoNE digit-wise Cross Entropy loss.
        
        Parameters
        ----------
        hidden_state : (N, n_embd) — The model's hidden states at [NUM] positions
        label        : (N,) float32 — The target numeric values
        """
        hidden_norm = self.layer_norm(hidden_state.float())
        num_digits = self.int_digit_len + self.frac_digit_len
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0] # Typically 4
        
        start_idx = 0
        end_idx = num_digits * num_cos_sin_per_digit
        
        slices_all_digits = hidden_norm[:, start_idx:end_idx]
        slices_all_digits = slices_all_digits.view(-1, num_digits, num_cos_sin_per_digit)
        
        logits_all_digits = torch.matmul(slices_all_digits, self.precomputed_cos_sin_matrix)
        
        # Get ground truth digits
        # scale so the lowest fractional digit is the units digit
        scaled_labels = torch.round(label * (10 ** self.frac_digit_len)).long()
        # extract digit for each power of 10
        digit_labels = (scaled_labels.unsqueeze(1) // self.powers_of_ten) % 10
        
        loss_per_digit = F.cross_entropy(
            logits_all_digits.view(-1, 10),
            digit_labels.view(-1),
            reduction='none'
        )
        
        return loss_per_digit.mean()
        
    def decode(self, hidden_state: torch.Tensor) -> float:
        """
        Decode a predicted number from the hidden state using argmax classification.
        """
        hidden_norm = self.layer_norm(hidden_state.float().unsqueeze(0)) # (1, n_embd)
        
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0]
        predicted_number = 0.0
        
        # Fractional digits
        for i in range(self.frac_digit_len):
            start_idx = i * num_cos_sin_per_digit
            end_idx = start_idx + num_cos_sin_per_digit
            slice_i = hidden_norm[:, start_idx:end_idx]
            logits_i = torch.matmul(slice_i, self.precomputed_cos_sin_matrix)
            predicted_digit = torch.argmax(logits_i, dim=1).item()
            predicted_number += predicted_digit * (10 ** -(self.frac_digit_len - i))
            
        # Integer digits
        for j in range(self.int_digit_len):
            start_idx = (self.frac_digit_len + j) * num_cos_sin_per_digit
            end_idx = start_idx + num_cos_sin_per_digit
            slice_j = hidden_norm[:, start_idx:end_idx]
            logits_j = torch.matmul(slice_j, self.precomputed_cos_sin_matrix)
            predicted_digit = torch.argmax(logits_j, dim=1).item()
            predicted_number += predicted_digit * (10 ** j)
            
        return round(predicted_number, self.frac_digit_len)

# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing FoNEEncoder ...")

    encoder = FoNEEncoder(n_embd=768, int_digit_len=7, frac_digit_len=2)
    print(f"  FoNE dim    : {encoder.fone_dim}  (should be 36 for 9 digits and 2 bases)")
    print(f"  n_embd      : {encoder.n_embd}    (768)")

    # Test encode 
    test_vals = torch.tensor([0.0, 1.0, 42.0, 1874.0, 123456.0, 3.14])
    embs = encoder.encode(test_vals)
    print(f"  encode shape: {embs.shape}  (should be [6, 768])")

    # Note: we can't do a perfect encode->decode roundtrip because the encoder 
    # uses a random linear projection, while the decoder expects the loss-supervised
    # un-scrambled Fourier features. To test decoder, we'd bypass the linear layer
    # or create dummy un-scrambled features.

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
