# Comparative Evaluation of Numerical Representations for Language Models: Digit Tokenization vs. Fourier Number Embeddings (FoNE) on BigBench Arithmetic

**Author:** AI Research Assistant & Student  
**Dataset:** BigBench Arithmetic (Mathematical Reasoning Benchmark)  
**Date:** July 2026  

---

## Executive Summary

This report presents a rigorous comparative analysis of two contrasting approaches for representing numerical quantities in Transformer-based Autoregressive Language Models: **Digit-level Tokenization** and **Fourier Number Embeddings (FoNE)**. 

Evaluated on the challenging **BigBench Arithmetic benchmark** (4,000 training examples, 500 validation examples, 500 test examples), **FoNE outperforms Digit Tokenization by a massive margin of +34.6 percentage points** in Exact Match accuracy (a **346% relative improvement**). 

| Model Representation | Exact Match Accuracy (EM) | Near Accuracy (±1% Tolerance) | Best Dev Loss | Convergence Epoch |
| :--- | :---: | :---: | :---: | :---: |
| **Digit Tokenization** | **10.0%** (50 / 500) | **17.2%** (86 / 500) | 0.9324 | Epoch 40 (Stopped Ep 57) |
| **Fourier Number Embeddings (FoNE)** | **44.6%** (223 / 500) | **51.6%** (258 / 500) | **0.4951** | Epoch 60 (Stopped Ep 77) |
| **Performance Difference** | **+34.6% (+346% rel)** | **+34.4% (+200% rel)** | **-0.4373 (Lower)** | +20 Epochs |

---

## 1. Task & Dataset Specification

### 1.1 Problem Formulation
The goal of this task is autoregressive mathematical reasoning: given a prompt specifying a multi-digit arithmetic problem (such as addition, subtraction, or multiplication of positive and negative integers), the model must generate the exact numerical solution.

Example prompt:
```
Input  : What is 757 minus 550?
Target : 207
```

### 1.2 Dataset Partitioning
The experiments were conducted on the BigBench Arithmetic benchmark using identical data splits across both models to eliminate variance:
* **Training Set:** 4,000 synthetic multi-digit arithmetic problems
* **Validation Set (Dev):** 500 held-out problems evaluated every 3 epochs
* **Test Set:** 500 unseen problems for final benchmark evaluation

---

## 2. Model Architecture & Representation Strategies

Both models share an identical GPT-2 Transformer backbone (12 layers, 12 attention heads, hidden dimension $d_{\text{model}} = 768$). The fundamental difference lies in how numerical values are encoded into token sequences and embeddings.

```
                      ┌─────────────────────────────────────────┐
                      │          Input: "What is 87 plus 684?"  │
                      └────────────────────┬────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
   ┌──────────────────────────┐                        ┌──────────────────────────┐
   │    Digit Tokenization    │                        │      FoNE Embedding      │
   ├──────────────────────────┤                        ├──────────────────────────┤
   │ "8 7" → ["8", "7"]       │                        │ "87" → [NUM] + Fourier   │
   │ "684" → ["6", "8", "4"]  │                        │ "684" → [NUM] + Fourier  │
   │ Tokens per num: N        │                        │ Tokens per num: 1        │
   │ Loss: Standard CE        │                        │ Loss: CE + Fourier CE    │
   │ Decode: Token-by-token   │                        │ Decode: Parallel Argmax  │
   └──────────────────────────┘                        └──────────────────────────┘
```

### 2.1 Digit Tokenization Model
* **Mechanism:** Numbers are explicitly split character-by-character into individual single-digit tokens before entering the tokenizer (e.g., `"1874"` $\rightarrow$ `["1", "8", "7", "4"]`).
* **Vocabulary:** 10 dedicated tokens (`0` through `9`) guaranteed in the vocabulary.
* **Loss Function:** Standard Cross-Entropy Language Modeling loss ($L_{\text{CE}}$) over token sequences.
* **Decoding:** Autoregressive token-by-token generation. Decoding a 4-digit number requires 4 sequential transformer generation steps.
* **Post-processing:** Generated digit tokens are rejoined (`"1 9 8"` $\rightarrow$ `"198"`) before string/numeric evaluation.

### 2.2 Fourier Number Embedding (FoNE) Model
* **Mechanism:** Every number (regardless of digit length or scale) is mapped to **1 single special token** `[NUM]`.
* **Embedding Injection:** At `[NUM]` positions, a 20-dimensional continuous Fourier feature vector $\Phi(x)$ is computed across multiple modular periods (bases of 10) and added directly to the token embedding:
  $$\mathbf{x}_t = \text{Embedding}(\text{[NUM]}) + \text{LayerNorm}(\mathbf{W}_{\text{linear}} \cdot \Phi(x)) + \text{Positional}(t)$$
* **Loss Function:** Joint multi-task objective:
  $$L_{\text{total}} = L_{\text{CE}} + \lambda L_{\text{FoNE\_digit}}$$
  where $L_{\text{FoNE\_digit}}$ applies digit-wise Cross-Entropy loss over the projected Fourier hidden state $\mathbf{h}_t$ to predict digit values across 7 integer and 2 fractional digit positions simultaneously.
* **Decoding:** Single-step generation. The model outputs a single `[NUM]` token. The hidden vector $\mathbf{h}_t$ is multiplied by the precomputed Fourier matrix $\mathbf{M}_{\text{fourier}}$, and an argmax across all digit positions reconstructs the entire number in parallel.

---

## 3. Experimental & Training Setup

Both models were trained using Distributed Data Parallel (DDP) across 4 NVIDIA GPUs.

* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay = $0.1$)
* **Learning Rate Schedule:** Cosine decay with linear warmup (Peak $\text{LR} = 1.0 \times 10^{-4}$)
* **Effective Batch Size:** $256$ ($64$ per GPU $\times 4$ GPUs)
* **Early Stopping:** Patience = 5 evaluations (checked every 3 epochs). Training terminates when validation loss fails to improve for 5 consecutive evaluations.
* **Decoding Strategy:** Greedy decoding ($\text{temperature} = 0.0$, $\text{repetition\_penalty} = 1.0$) for exact deterministic mathematical evaluation.

---

## 4. Empirical Results & Performance Analysis

### 4.1 Test Set Accuracy

Evaluation was performed on the 500 unseen BigBench test questions using two standardized metrics:
1. **Exact Match Accuracy (EM):** Generated number exactly equals target number ($x_{\text{pred}} == x_{\text{target}}$).
2. **Near Accuracy (±1% Tolerance):** Generated number is within 1% relative error of the target ($|x_{\text{pred}} - x_{\text{target}}| / |x_{\text{target}}| \le 0.01$).

```
Exact Accuracy (EM) %
FoNE    : ■■■■■■■■■■■■■■■■■■■■■■■■■■ 44.6%
Digit   : ■■■■■ 10.0%

Near Accuracy (±1%) %
FoNE    : ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 51.6%
Digit   : ■■■■■■■■■ 17.2%
```

| Metric | Digit Model | FoNE Model | Absolute Gain | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Match (EM)** | 10.0% (50/500) | **44.6%** (223/500) | **+34.6%** | **+346%** |
| **Near Match (±1%)** | 17.2% (86/500) | **51.6%** (258/500) | **+34.4%** | **+200%** |

### 4.2 Training Convergence Dynamics

Analysis of training logs (`logs/digit_bigbench_train.log` vs `logs/fone_bigbench_train.log`):

```
Validation Loss Comparison over Epochs:
Epoch 04 : Digit = 2.3895  |  FoNE = 1.0506
Epoch 10 : Digit = 1.2302  |  FoNE = 0.8540
Epoch 20 : Digit = 1.0309  |  FoNE = 0.6736
Epoch 40 : Digit = 0.9324* |  FoNE = 0.5160
Epoch 60 : Digit = Stopped |  FoNE = 0.4951*
```
*\* Denotes epoch where lowest validation loss was achieved.*

1. **Loss Floor:** FoNE achieves a substantially lower minimum validation loss (**0.4951**) compared to Digit Tokenization (**0.9324**), reflecting far lower cross-entropy perplexity on numerical sequences.
2. **Convergence Trajectory:** 
   * **Digit Model:** Rapid initial loss drop, but plateaus around Epoch 37–40 (`dev_loss` ~0.93–0.95) and triggers early stopping at Epoch 57.
   * **FoNE Model:** Steady, continuous optimization over 60 epochs (`dev_loss` improving down to 0.4951), triggering early stopping at Epoch 77.

---

## 5. Qualitative Error & Behavioral Analysis

Examining individual predictions from `results/bigbench_digit-final.json` and `results/bigbench_fone-final.json`:

### Sample Comparison Table

| Example ID | Prompt Input | True Target | Digit Model Prediction | FoNE Model Prediction | Digit Status | FoNE Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `arith_00000` | What is 87 plus 684? | **771** | `844` | **`771`** | ❌ Fail | ✅ Correct |
| `arith_00001` | What is 757 minus 550? | **207** | `185` | **`207`** | ❌ Fail | ✅ Correct |
| `arith_00002` | What is 427 minus 246? | **181** | **`181`** | **`181`** | ✅ Correct | ✅ Correct |
| `arith_00003` | What is 82 times 38? | **3116** | `2954` | **`3116`** | ❌ Fail | ✅ Correct |

---

## 6. Key Scientific & Architectural Insights

Why does FoNE outperform Digit Tokenization by such a significant margin on mathematical arithmetic?

### 1. Autoregressive Error Cascade Elimination
* **Digit Model:** To output `3116`, the Digit model must perform 4 sequential autoregressive decisions (`"3"` $\rightarrow$ `"1"` $\rightarrow$ `"1"` $\rightarrow$ `"6"`). If the model makes an error on the most significant digit (e.g. predicting `"2"` instead of `"3"`), the subsequent digits are conditioned on a corrupted prefix and cannot recover.
* **FoNE Model:** FoNE generates a single `[NUM]` token. The entire number (`3116`) is decoded in parallel from the hidden state $\mathbf{h}_t$. This completely eliminates autoregressive error accumulation for numerical outputs.

### 2. Sequence Length & Attention Complexity
* **Digit Model:** Splitting numbers into digits increases sequence length by 3x–5x. Self-attention complexity scales quadratically ($O(N^2)$), diluting attention weights between operators (`plus`, `minus`, `times`) and operands.
* **FoNE Model:** Every number occupies exactly 1 token position, keeping sequence lengths short and dense. Attention heads can directly attend from operator to operand without navigating multi-token sub-strings.

### 3. Continuous Fourier Periodic Encoding
* **Digit Model:** Treats digits `0` through `9` as discrete categorical tokens. The model has no intrinsic inductive bias that digit `7` is quantitatively closer to `8` than to `2`.
* **FoNE Model:** Uses sinusoidal Fourier basis functions across scale periods ($10^{-2}, 10^{-1}, 10^0, 10^1, \dots$). This imparts a strong mathematical inductive bias, allowing the network's continuous geometry to represent magnitude smoothly.

---

## 7. Conclusion & Recommendations

### Summary of Findings
1. **Fourier Number Embeddings (FoNE)** provide a dramatically superior representation for mathematical reasoning compared to character-level **Digit Tokenization**, achieving **44.6% EM vs 10.0% EM** (+346% relative gain) on BigBench Arithmetic.
2. FoNE converges to a much lower loss floor (0.4951 vs 0.9324) and avoids the severe autoregressive error cascade inherent in multi-token digit generation.

### Recommendation for Future Work
* **Scaling Data:** Expand training set size beyond 4,000 examples to test if FoNE can reach >80% accuracy on complex multi-digit arithmetic.
* **Evaluation on General Text-Table Tasks:** Evaluate FoNE on broader numerical reasoning benchmarks (such as ToTTo and FinQA) where numerical values appear alongside natural language context.

---
*Report generated for research synthesis and academic presentation.*
