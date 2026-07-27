"""
model/gpt2_digit.py
====================
GPT-2 with Digit Tokenization (no architectural changes).

The ONLY difference from GPT2BPE is the tokenizer:
  - BPE model   : "1874" → possibly ["18", "74"] or ["1874"] depending on BPE merges
  - Digit model : "1874" → ["1", "8", "7", "4"] — one token per digit

This gives the model direct access to the digit-level structure of numbers.
Every digit 0-9 is guaranteed a unique, dedicated token in the vocabulary.

Architecture: IDENTICAL to GPT2BPE. No embedding changes, no extra heads.
The improvement (if any) comes entirely from the tokenization strategy.

Usage
-----
  from model.gpt2 import GPT2Config
  from model.gpt2_digit import GPT2Digit

  config = GPT2Config(vocab_size=digit_vocab_size)
  model  = GPT2Digit(config)
  loss, logits = model(input_ids, labels=labels)
"""

# GPT2Digit is architecturally identical to GPT2BPE.
# We subclass it purely for naming / checkpoint clarity.

from model.gpt2_bpe import GPT2BPE
from model.gpt2 import GPT2Config


class GPT2Digit(GPT2BPE):
    """
    GPT-2 with digit-level tokenization.

    Architecturally identical to GPT2BPE. The digit-level structure is
    entirely in the tokenizer; this model class exists only to:
      1. Make it clear which tokenizer to use (totto_digit/ vs totto_bpe/)
      2. Produce distinct checkpoint filenames (digit_best.pt etc.)
      3. Allow a different vocab_size when the digit tokenizer has a
         different vocabulary from the BPE tokenizer.

    Parameters
    ----------
    config : GPT2Config — use the digit tokenizer's vocab_size here
    """

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GPT2Digit("
            f"n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
            f"n_embd={cfg.n_embd}, vocab={cfg.vocab_size}, "
            f"params={self.num_parameters()/1e6:.1f}M)"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("Building GPT2Digit model ...")
    config = GPT2Config(
        vocab_size  = 16000,
        n_positions = 512,   # digit sequences can be longer
        n_layer     = 12,
        n_head      = 12,
        n_embd      = 768,
        dropout     = 0.1,
    )
    model = GPT2Digit(config)
    print(model)
    print(f"Parameters: {model.num_parameters()/1e6:.2f}M")

    B, T  = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels    = input_ids.clone()
    labels[:, :50] = -100

    loss, logits = model(input_ids, labels=labels)
    print(f"\nForward pass:")
    print(f"  loss   : {loss.item():.4f}")
    print(f"  logits : {logits.shape}")

    print("\n✓ GPT2Digit OK")
