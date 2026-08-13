# Option B: Detached Numeric Projection Head (FoNE-DP)

## Background

Both xVal and FoNE suffer from **Representation Conflict**: the hidden state at `[NUM]` positions is pulled in two directions simultaneously — the CE loss wants it to encode linguistic context (to predict the next word), and the numeric loss wants it to encode numeric information (magnitude, digits). This overwrites the linguistic signal and causes hallucination loops and garbage number generation.

**Option B (our fix):** Keep the GPT-2 backbone completely unchanged at 768 dims. Add a small numeric projection head (`768 → 64 → 768`) that branches off the hidden state at `[NUM]` positions. The head produces a 768-dim vector that is fed into the existing `FoNEEncoder.compute_loss()` for digit-wise Cross-Entropy supervision. Critically, we **`detach()`** the hidden state before passing it to this head, so FoNE digit-CE gradients **cannot flow back into the backbone**. The backbone is trained solely by CE loss. At generation time, the projected vector is decoded via `FoNEEncoder.decode()` to recover the number value.

We call this model **FoNE-DP** (FoNE + Detached Projection).

> **Why FoNE's digit-CE over xVal's scalar MSE?**
> FoNE's digit-wise CE is a better-calibrated loss for numbers. It supervises each decimal digit independently (ones, tens, hundreds, …) via Fourier features, giving gradient signal for each position rather than a single collapsed scalar. This means the projection head learns a structured, digit-decomposed representation — much easier to learn from 768 input dims than predicting a raw float with MSE.

---

## Addressing All 5 Nuances

| # | Nuance | Strategy in xVal-DP |
|---|--------|---------------------|
| **a** | Input-side scaling (large value bombs) | FoNE-DP **does not multiply embeddings by number values** (that is xVal's approach). FoNE instead **adds** a Fourier-encoded vector to the `[NUM]` token embedding. This additive signal is bounded by construction (cos/sin values ∈ [-1, 1] scaled by a learned linear). No blowup risk. We keep FoNE's standard additive input encoding unchanged. |
| **b** | Positional encoding at `[NUM]` | Not a problem: we keep `n_embd=768` throughout. No dimension expansion, so no new PE dims to worry about. |
| **c** | LayerNorm mixing | Not a problem for same reason — no dimension expansion. The numeric projection subspace only exists inside the detached head, never in the residual stream. |
| **d** | Attention over `[NUM]` tokens | Natural behavior: attention from future tokens to `[NUM]` is handled through the 768-dim hidden state as usual. The projected numeric representation is only used for the digit-CE loss output, not re-injected into the residual stream. |
| **e** | Training stability (gradient magnitude mismatch) | `detach()` completely isolates the two losses. FoNE digit-CE gradients only update the `num_proj` MLP weights (~590K params), not the 85M backbone at all. The `num_loss_weight` hyperparameter only affects the projection head training. |

---

## Design Decisions

### Input-Side Encoding: Keep FoNE's Additive Approach (Nuance a)
FoNE's input encoding adds a Fourier vector to the `[NUM]` embedding: `x[t] = wte([NUM]) + FoNE(value) + wpe(t)`. This additive signal is inherently bounded (sin/cos features in `[-1, 1]`, then projected by a learned `Linear(768, 768)`). This is **much safer** than xVal's multiplicative scaling. We keep this standard FoNE input encoding unchanged.

### Numeric Projection Head
Instead of a tiny scalar head, we use a **768-dim bottleneck** so the FoNE decoder (which expects 768-dim input with structured Fourier layout in its first `(int_digit_len + frac_digit_len) * 4` dims) can actually read from it:

```python
num_proj = nn.Sequential(
    nn.Linear(768, 64),    # bottleneck: compress to 64 dims
    nn.GELU(),
    nn.Linear(64, 768),    # expand back to 768 dims for FoNE decode
)
# Total new params: 768*64 + 64 + 64*768 + 768 = ~98,560 params (0.12% of model)
```

The output of `num_proj` is a 768-dim vector in the **same space as the FoNE decoder expects**. The existing `FoNEEncoder.compute_loss(proj_out, targets)` and `FoNEEncoder.decode(proj_out)` methods work without modification.

### Gradient Isolation
```python
h_detached = hidden.detach()          # FoNE digit-CE gradient stops here
proj_out   = num_proj(h_detached[num_mask])   # (N, 768)
fone_loss  = fone_encoder.compute_loss(proj_out, true_values)   # digit-wise CE
```

The `detach()` means:
- **CE backward pass:** flows through backbone normally, updates all 85M params
- **FoNE digit-CE backward pass:** flows through `num_proj` only (98K params), **wall at `detach()`**

### Generation: Decode via FoNE
At generation time, for each `[NUM]` token produced:
```python
last_hidden = hidden[:, -1, :]           # (B, 768) — no detach needed (no grad)
proj_out    = num_proj(last_hidden)       # (B, 768)
for b in range(B):
    if next_token[b] == num_token_id:
        pred_val = fone_encoder.decode(proj_out[b])  # digit-by-digit argmax → float
        predicted_values[b].append(pred_val)
```

---

## Files to Create / Modify

### Model Layer

#### [NEW] model/gpt2_fone_dp.py
New model class `GPT2FoNeDP` ("FoNE with Detached Projection").

Key changes vs `gpt2_fone.py`:
1. Add `num_proj = nn.Sequential(Linear(768, 64), GELU, Linear(64, 768))` — a 768→64→768 bottleneck projection head
2. In `forward()`: hidden state at `[NUM]` label positions is **detached** before being passed through `num_proj`, then into `fone.compute_loss()`
3. The **backbone receives no FoNE digit-CE gradients at all**; it is trained solely by CE loss
4. In `generate()`: `num_proj(last_hidden)` produces the 768-dim vector that `fone.decode()` reads digit-by-digit via argmax; no `detach()` needed here (wrapped in `torch.no_grad()`)
5. The FoNE **input encoding** (`encode_sequence` at [NUM] positions) is kept **unchanged** — the additive FoNE embedding still informs the backbone about number magnitude on the input side

---

### Training Layer

#### [NEW] train/train_fone_dp.py
Copy of `train_fone.py` with:
- Import `GPT2FoNeDP` instead of `GPT2FoNE`
- `run_name = "fone_dp"`
- Uses same `fone_collate_fn` and `ToTToDatasetFoNE` — **no data changes**
- `step_fn` returns `ce_loss`, `fone_loss`, `total_loss` for logging

---

### Config Layer

#### [NEW] configs/model_config_fone_dp.yaml
Copy of `model_config.yaml` with additions:
```yaml
# FoNE-DP specific
num_proj_hidden: 64     # bottleneck dim in the detached projection head
num_loss_weight: 1.0    # weight for FoNE digit-CE loss (trains proj head only)
int_digit_len:   7      # max integer digits (same as standard FoNE)
frac_digit_len:  2      # max fractional digits
```

---

### Eval Layer

#### [NEW] eval/eval_fone_dp.py
Copy of `eval_fone.py` with:
- Import `GPT2FoNeDP`
- Load the FoNE-DP checkpoint
- At `[NUM]` positions, call `num_proj(last_hidden)` then `fone.decode()` to recover number value
- Output saved to `results/fone_dp_totto-200.json`

---

## Verification Plan

### 1. Unit Test (model forward pass)
```bash
python model/gpt2_fone_dp.py
```
Verifies:
- **Gradient isolation test:** run a backward pass, then assert that `model.fone.linear.weight.grad` is `None` (no FoNE-CE gradient reached the backbone's FoNE input encoder), while `model.num_proj[0].weight.grad` is non-None (the projection head did get trained)
- Assert `model.wte.weight.grad` is not contaminated by FoNE digit-CE gradients
- Generation produces a plausible float from `fone.decode(num_proj(last_hidden))`

### 2. Training
```bash
torchrun --nproc_per_node=4 train/train_fone_dp.py \
    --config configs/train_config.yaml \
    --model_config configs/model_config_fone_dp.yaml
```
Monitor `ce_loss` and `fone_loss` separately in logs. Expect `fone_loss` to converge independently without affecting `ce_loss` curve.

### 3. Evaluation (same 200 examples as existing results)
```bash
python eval/eval_fone_dp.py \
    --checkpoint checkpoints/fone_dp/best.pt \
    --config configs/model_config_fone_dp.yaml \
    --train_config configs/train_config.yaml \
    --output results/fone_dp_totto-200.json \
    --max_examples 200
```

### 4. Visual comparison
Add `fone_dp` as a 5th model to `app-simple.py` to directly compare outputs side-by-side with the original FoNE.

---

## Is this publishable at an A* conference?

**Honest assessment: Not as a standalone contribution, but yes as part of a larger story.**

What's strong about this:
- Clearly diagnoses a real problem (representation conflict) with mechanistic evidence from the garbage-output results of both FoNE and xVal
- FoNE-DP is a minimal, ablation-clean fix (adds ~0.12% parameters — a 768→64→768 projection head)
- The `detach()` trick makes the comparison surgical: the only variable is whether numeric loss gradients reach the backbone
- FoNE's digit-wise CE is a better-grounded numeric supervision than xVal's scalar MSE — this strengthens the model design argument
- Systematic evaluation across two datasets with quantitative metrics

What's missing for A* (ACL/EMNLP/NeurIPS/ICLR):
1. **Scale:** A* conferences expect experiments on larger models (GPT-2 medium at minimum, ideally 1B+)
2. **Breadth:** Need 3+ datasets; you have 2
3. **Baselines:** Should compare against number tokenization papers, ProofWriter, etc.
4. **Theoretical grounding:** The "representation conflict" hypothesis should be verified quantitatively — e.g., measure the cosine similarity between CE-gradient direction and FoNE-digit-CE-gradient direction at `[NUM]` positions is negative (confirming they conflict), then show it becomes ≈0 after applying `detach()`
5. **Ablations:** `num_loss_weight` sensitivity, `num_proj_hidden` dim sensitivity, with vs. without detach (i.e., standard FoNE as the ablation), with vs. without FoNE input encoding

**Realistic target:** A strong workshop paper (NeurIPS workshop on Numerical Reasoning, or ACL Findings) or ARXiV preprint. If results are dramatic (FoNE hallucination loops → fluent + accurate numbers), it becomes a compelling short paper at a B conference (EMNLP Findings, COLING).

The core diagnostic + fix narrative is publication-worthy at the right venue. The key is whether FoNE-DP eliminates hallucination loops while matching/exceeding the original FoNE on numeric EM — that's the make-or-break experiment.
