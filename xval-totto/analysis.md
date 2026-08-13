# Comprehensive Model Analysis: ToTTo (n=200) and BigBench (n=500)

> **Scope:**
> - **ToTTo (Table-to-Text)**: 200 dev examples per model — **FINAL BENCHMARK (n=200)**
> - **BigBench Arithmetic**: 500 test examples per model — **FINAL BENCHMARK (n=500)**
> All 4 models (BPE, Digit, FoNE, xVal) evaluated across both tasks.

---

## 1. Summary Tables & Cross-Task Performance Matrix

### 1a. ToTTo — Table-to-Text Faithful Copying *(n=200 dev set, 179 contain numbers)*

| Model | BLEU ↑ | Num-EM ↑ | Hallucination Rate ↓ | Count-Match ↑ | Num-MAE ↓ |
|-------|--------|----------|----------------------|---------------|-----------|
| **BPE** | **33.41** | **0.9064** (90.6%) | **0.1056** (10.6%) | **0.7000** (70.0%) | 126,703.19 |
| **Digit** | 25.88 | 0.7245 (72.5%) | 0.2632 (26.3%) | 0.6162 (61.6%) | **14,136.65** |
| **FoNE** | 18.41 | 0.7224 (72.2%) | 0.5348 (53.5%) | 0.2600 (26.0%) | $3.16 \times 10^{129}$ * |
| **xVal** | 15.12 | 0.2127 (21.3%) | 0.8200 (82.0%) | 0.3300 (33.0%) | 10,458.71 |

*\* FoNE Num-MAE is inflated by runaway unconstrained number concatenation on multi-number sentences.*  
*\* Digit model strictly skipped 2 examples that exceeded the 256 token context limit.*

### 1b. BigBench Arithmetic — Mathematical Reasoning *(n=500 test set)*

| Model | Exact Acc ↑ | Near Acc (±1%) ↑ | Key Observation |
|-------|-------------|------------------|-----------------|
| **FoNE** | **0.446** (44.6%) | **0.516** (51.6%) | Dedicated regression head provides strong arithmetic inductive bias |
| **Digit** | 0.100 (10.0%) | 0.172 (17.2%) | Autoregressive digit prediction suffers from compounding errors |
| **BPE** | 0.020 (2.0%) | 0.056 (5.6%) | Generates plausible-magnitude numbers, but cannot perform computation |
| **xVal** | 0.002 (0.2%) | 0.022 (2.2%) | Continuous scalar output rarely lands on exact integer targets |

### 1c. Complete 2×4 Task Comparison Matrix

| Model | ToTTo BLEU ↑ | ToTTo Num-EM ↑ | BigBench Exact Acc ↑ | Primary Strengths & Weaknesses |
|-------|--------------|----------------|----------------------|--------------------------------|
| **BPE** | **33.41** | **0.9064** | 0.0200 | 🟢 Best at text fluency & copying; 🔴 Cannot compute arithmetic |
| **Digit** | 25.88 | 0.7245 | 0.1000 | 🟡 Balanced middle-ground; 🔴 Longer sequence lengths & digit drift |
| **FoNE** | 18.41 | 0.7224 | **0.4460** | 🟢 Dominates arithmetic; 🔴 High hallucination & poor BLEU on text |
| **xVal** | 15.12 | 0.2127 | 0.0020 | 🔴 Fails on exact values; continuous scalar head degrades LM decoding |

---

## 2. In-Depth Technical Analysis of Results

### 2a. Why BPE Dominates Table-to-Text (ToTTo: 90.6% Num-EM, 33.4 BLEU)
- **Opaque Subword Copying**: In table-to-text generation, numbers appear in the table header/cells and must be reproduced verbatim in the sentence. Standard BPE represents whole numbers (e.g. `1874`, `2016`) as single vocabulary tokens.
- **Attention Alignment**: Self-attention layers easily learn direct copy operations from input subword tokens to target subword tokens in a single generation step.
- **Low Hallucination (10.6%)**: Because BPE emits complete subword tokens, it rarely produces fragments or spurious floating-point numbers.

### 2b. Why FoNE Dominates Arithmetic (BigBench: 44.6% Exact Acc), but Struggles on Text
- **Supervised Regression Head**: FoNE's `[NUM]` token architecture uses a dedicated scalar regression head trained directly on target numbers. This continuous numeric supervision provides a strong inductive bias for arithmetic operations (addition, subtraction).
- **The Text Generation Drawback**: When generating natural language (ToTTo), supervising hidden states for numerical regression distorts the transformer's language modeling distribution. On multi-number sequences, FoNE frequently suffers from **runaway concatenation** (emitting repeated `[NUM]` tokens or unbounded floating-point literals like `1.201e+04`), resulting in a **53.5% hallucination rate** and low BLEU (18.41).

### 2c. Why Digit Model Performs Moderately Across Both Tasks (72.5% Num-EM, 10.0% Exact Acc)
- **Digit Granularity**: Splitting numbers into individual space-separated digits (`1 8 7 4`) forces the model to attend to individual digits.
- **Compounding Autoregressive Errors**: Generating a 4-digit number requires 4 consecutive correct next-token predictions. An error on any single digit invalidates the entire number.
- **Context Overhead**: Digit-split sequences are 2-3× longer than BPE, causing 2 examples to exceed the 256 context window limit.

### 2d. Why xVal Fails on Exact Evaluation (21.3% Num-EM, 0.2% Exact Acc)
- **Log-Normalized Continuous Representation**: xVal scales the embedding vector of `[NUM]` by the log-normalized magnitude of the number and uses a scalar head to predict outputs.
- **Continuous vs. Discrete Mismatch**:
  - On **BigBench**, exact-match evaluation requires integer targets. xVal outputs continuous float predictions (e.g. `151.6` for target `207`), which almost never satisfy exact match ($0.2\%$).
  - On **ToTTo**, continuous scalar embeddings interfere with the language model head during decoding. When generating text after a `[NUM]` token, xVal often gets trapped in **numeric hallucination loops** (emitting 10-20 continuous float numbers in a row), driving its hallucination rate to **82.0%** and BLEU down to **15.12**.

---

## 3. Key Theoretical Takeaways

1. **The Representation Trade-Off**: No single numerical representation method excels at both faithful copying and arithmetic computation:
   - **Discrete Subwords (BPE)** represent numbers as opaque symbols $\rightarrow$ optimal for **faithful copying / extraction**.
   - **Continuous Regression Heads (FoNE)** represent numbers as scalar values $\rightarrow$ optimal for **mathematical computation**.

2. **Continuous Embeddings Degrade Autoregressive Text Generation**: Scaling token embeddings or supervising hidden states with continuous values disrupts the standard softmax probability distribution of language models during free-form text generation.

3. **Evaluation Metric Sensitivity**: Continuous methods (xVal, FoNE) require task-appropriate loss functions and evaluation metrics. Exact integer match severely penalizes continuous representation models like xVal.

---

## 4. Conclusion & Recommendations

- **For Information Extraction / Table-to-Text**: Standard **BPE tokenization** remains the most effective and reliable choice.
- **For Dedicated Mathematical / Scientific Computation**: **FoNE (Fourier Number Embeddings)** provides a superior architecture for numerical reasoning.
- **Future Direction**: Hybrid architectures that maintain discrete subword tokens for language generation while routing numeric calculations to a dedicated FoNE/arithmetic module represent the most promising path forward.
