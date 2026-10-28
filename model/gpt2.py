"""
model/gpt2.py
=============
Shared GPT-2 building blocks used by BOTH models.

Contains:
  - GPT2Config      : dataclass with all architectural hyperparameters
  - CausalSelfAttention : multi-head attention with causal mask
  - MLP             : feed-forward block
  - TransformerBlock: one transformer layer (pre-norm residual)
  - GPT2Backbone    : stacked transformer layers (no embedding, no head)

Neither GPT2BPE nor GPT2XVal modify anything in this file.
The ONLY differences between the two models live in gpt2_bpe.py and
gpt2_xval.py (the embedding step and the output head).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class GPT2Config:
    """
    All architectural hyperparameters for the GPT-2 model.

    Both BPE and xVal models use the same config object.
    Load from configs/model_config.yaml in training scripts.
    """
    vocab_size:  int   = 16000   # size of the BPE tokenizer vocabulary
    n_positions: int   = 256     # maximum sequence length (context window)
    n_layer:     int   = 12      # number of transformer blocks
    n_head:      int   = 12      # number of attention heads
    n_embd:      int   = 768     # embedding dimension (d_model)
    dropout:     float = 0.1     # dropout probability (attn + residual)
    bias:        bool  = True    # use bias in Linear layers and LayerNorm


# ── Attention ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+),
    which automatically uses Flash Attention when available on the GPU.
    This is significantly faster and more memory-efficient than the naive
    manual implementation, especially for longer sequences.

    Causal masking is handled by is_causal=True — no explicit mask tensor
    needs to be constructed or stored.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        # Project input to Q, K, V all at once (3× the embedding dimension)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Project concatenated heads back to embedding dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, n_embd)  — input hidden states

        Returns
        -------
        (B, T, n_embd)  — output hidden states after self-attention
        """
        B, T, C = x.size()

        # ── Compute Q, K, V ───────────────────────────────────────────────────
        # c_attn maps (B, T, n_embd) → (B, T, 3*n_embd)
        # then split along last dim into three (B, T, n_embd) tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── Flash Attention ───────────────────────────────────────────────────
        # is_causal=True: token at position i can only attend to positions 0..i
        # dropout_p: attention dropout (only applied during training)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # ── Merge heads and project ───────────────────────────────────────────
        # (B, n_head, T, head_dim) → (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.

    Hidden dimension is 4× the embedding dimension (standard GPT-2).
    Pre-norm is applied outside this module (in TransformerBlock).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One GPT-2 transformer layer using pre-normalization.

    Pre-norm means LayerNorm is applied to the INPUT of each sub-block,
    then the result is added back to the ORIGINAL (residual connection).
    This is the architecture used in GPT-2 and most modern LLMs.

    Pre-norm (used here):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Post-norm (original Transformer, Vaswani 2017):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + MLP(x))
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # attention sub-block (pre-norm)
        x = x + self.mlp(self.ln_2(x))    # MLP sub-block (pre-norm)
        return x


# ── Backbone ──────────────────────────────────────────────────────────────────

class GPT2Backbone(nn.Module):
    """
    The shared transformer backbone: stacked TransformerBlocks + final LayerNorm.

    This module does NOT include:
      - Token embedding (wte)
      - Position embedding (wpe)
      - Any output head (lm_head or scalar_head)

    It takes an already-embedded tensor (B, T, n_embd) and returns
    final hidden states (B, T, n_embd).

    BOTH GPT2BPE and GPT2XVal instantiate this identical backbone.
    The difference between the two models is entirely in the embedding
    step (before this module) and the output head (after this module).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.h     = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.drop  = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, n_embd) — embedded input (token + position embeddings)

        Returns
        -------
        (B, T, n_embd) — final hidden states (after n_layer blocks + LayerNorm)
        """
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Weight initialization helper ─────────────────────────────────────────────

def init_weights(module: nn.Module, n_layer: int) -> None:
    """
    Initialize weights following the GPT-2 paper.

    Standard init: Normal(0, 0.02) for Linear and Embedding weights.
    Residual projections (c_proj in attention and MLP): scaled down by
    sqrt(2 * n_layer) to prevent the residual stream from growing with depth.
    Zeros for biases, ones for LayerNorm scale.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # Special scaled initialization for residual projections
    # (applied on top of the above, overrides c_proj.weight specifically)
    for name, param in module.named_parameters():
        if name == "c_proj.weight":
            nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
