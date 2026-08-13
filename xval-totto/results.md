# Numerical Representations in Language Models
## Bridging Computation and Faithful Text Generation
### 10-Minute Research Presentation

---

## Slide 1 — Motivation and Central Question

**The Problem with Numbers in LLMs**

Standard BPE tokenizers are trained on character co-occurrence frequency. They treat numbers as character strings and apply merges greedily — with no awareness of numeric meaning:

- `"1874"` → `["18", "74"]` — magnitude relationship to "1873" is invisible
- `"23.5"` → `["23", ".", "5"]` — the decimal structure is fragmented
- `"12345"` → `["12", "345"]` — entirely data-dependent, non-deterministic

This destroys **magnitude**, **continuity**, and **digit structure** — the very properties that make numbers different from words.

**Our Central Question:**

> Do specialised number representations improve *both* arithmetic computation *and* faithful number generation from tables?

We compare **4 number representation strategies** on **2 complementary tasks** — this is a controlled comparison study, not a new-method paper.

---

## Slide 2 — The 4 Models: Overview

| Model | Core Idea | Technical Mechanism | Designed For |
|---|---|---|---|
| **BPE** | Numbers are subword tokens, treated like any word | Standard BPE vocabulary; next-token CE loss | General NLP |
| **Digit** | Each digit 0–9 gets its own token | Metaspace pre-tokenizer forces digit isolation before BPE; `rejoin_digit_tokens()` at eval | Arithmetic via digit-level prediction |
| **FoNE** | Replace numbers with `[NUM]`; embed using sine/cosine at multiple frequency scales | Fourier features per number; MSE regression head supervises value at `[NUM]` positions | Arithmetic |
| **xVal** | Replace numbers with `[NUM]`; scale the `[NUM]` embedding vector by the actual number | `emb(x) = x * e_[NUM]`; log-normalised; MSE scalar head | **Scientific time-series prediction** |

---

## Slide 3 — Technical Deep-Dive: BPE (Baseline)

**BPE (Byte-Pair Encoding)**

BPE learns a vocabulary of ~16,000 subword merges from the training corpus by iteratively merging the most frequent character pair. Numbers are treated as character strings with no special handling:

- `"1874"` → `["18", "74"]` if "18" and "74" are high-frequency bigrams in the corpus
- The same number can tokenize differently depending on what surrounds it

There is **no numeric supervision** — the model learns number representations purely from next-token cross-entropy loss.

**In our codebase:**
```python
# data/dataset_bpe.py — minimal preprocessing
text = linearized_table + " [SEP] " + target_sentence
input_ids = tokenizer.encode(text).ids   # standard BPE encoding
labels    = input_ids.copy()             # predict everything after [SEP]
```

**Why BPE can copy but cannot compute:**
The model sees `"1874"` as an opaque vocabulary item — exactly like `"London"`. Self-attention layers learn to perform **copy attention**: they align the source token `"1874"` with the corresponding position in the output. This works well because the same opaque token appears in both table and target.

However, there is no signal from which to learn that `"1874" + 1 = "1875"`. BPE has no internal representation of the number's magnitude.

---

## Slide 4 — Technical Deep-Dive: Digit Tokenization

**Digit Model**

A custom BPE tokenizer is trained on a digit-pre-split corpus. The `Metaspace` pre-tokenizer forces a **space before every digit character** before BPE merges run:

- Training text: `"from 1874"` → `"from ▁1 ▁8 ▁7 ▁4"` (Metaspace prepend)
- Since each digit is an atomic pre-token, BPE **cannot merge digits across number boundaries**
- Result: at inference, every digit 0–9 is guaranteed to be its own vocabulary token

**Key implementation:**
```python
# tokenizer/train_digit_tokenizer.py
pre_tokenizer = Metaspace(prepend_scheme="always")
# Digits become isolated pre-tokens; BPE merges cannot cross digit boundaries
```

**Post-processing at eval:** The decoder outputs `"1 8 7 4"` (space-separated digits). The function `rejoin_digit_tokens()` merges consecutive digit tokens back into `"1874"`.

**Important note — Bug we fixed:** The MetaspaceDecoder was attaching trailing punctuation to the last digit (e.g., `"4."` instead of `"4"` `"."`). This prevented rejoining. After fixing this, Digit's ToTTo Num-EM improved from **23.6% → 72.5%**.

**Why Digit partially computes:**
The model explicitly sees each digit character. It can learn digit-level patterns (e.g., that adding two 3-digit numbers produces a 3–4 digit result). But generating a 4-digit number requires **4 consecutive correct next-token predictions** — a single wrong digit invalidates the entire number.

---

## Slide 5 — Technical Deep-Dive: FoNE

**FoNE (Fourier Number Embeddings)**

All numbers in the text are replaced by the single special `[NUM]` token. The embedding for `[NUM]` is **not a standard lookup** — it is computed from the number's actual value using sinusoidal (Fourier) features:

```
emb(x) = [ sin(x/p₁), cos(x/p₁), sin(x/p₂), cos(x/p₂), ... ]
```

The periods `p₁, p₂, ...` are set to `1, 10, 100, 1000, ...` — one pair per decimal place of precision. This gives the model explicit information about the ones digit, tens digit, hundreds digit, etc.

**Training loss** adds MSE regression on `[NUM]` positions to the standard cross-entropy loss:
```
L = L_CE  +  λ * L_MSE
```

The model's hidden state at each `[NUM]` position is passed through a scalar projection head that must predict the correct numeric value.

**In our codebase:**
```python
# model/gpt2_fone.py
class FoNEEncoder:
    def encode_sequence(self, input_ids, number_values, number_mask):
        embeddings = base_embedding(input_ids)
        for t in [NUM] positions:
            embeddings[t] = self.fourier_encode(number_values[t])  # sin/cos
        return embeddings

self.scalar_head = nn.Linear(n_embd, 1)   # regression head for MSE loss
```

**Why FoNE excels at arithmetic:**
The Fourier basis directly encodes magnitude (via low-frequency components) and digit periodicity (via high-frequency components). The MSE regression loss provides an explicit numeric supervision signal — the model must internally represent the number's actual value to minimise this loss. This is precisely the kind of inductive bias needed for addition and subtraction.

---

## Slide 6 — Technical Deep-Dive: xVal

**xVal (Continuous Value-Scaled Embeddings)**

xVal also replaces numbers with the `[NUM]` token, but embeds them via a fundamentally different mechanism:

```
emb(x) = x · e_[NUM]
```

The `[NUM]` embedding vector is **scaled by the actual numeric value**. The model learns a single "numeric direction" in embedding space; the magnitude of the vector encodes the number's value.

Numbers are **log-normalised** before scaling, stabilising training across values spanning many orders of magnitude:
```python
normalized = sign(x) * log1p(|x|)
```

**In our codebase:**
```python
# model/gpt2_xval.py — the key forward-pass modification
scale = torch.where(
    number_mask,           # True at [NUM] positions
    number_values,         # multiply by log-normalised value
    torch.ones_like(...)   # multiply by 1.0 at all other positions
).unsqueeze(-1)            # broadcast over embedding dimension

tok_emb = tok_emb * scale  # only [NUM] tokens are scaled; text unchanged
```

**Original design intent — why this matters:**
xVal was designed by the **Polymathic AI** group (Golkar et al., 2023) for **scientific time-series prediction**. Its evaluation benchmarks were physics simulations and sensor readouts where:
- Inputs and outputs are **dense numerical sequences** (not free natural language text)
- The output is always **a number**, never a mixed text+number sentence
- Exact integer match is irrelevant — what matters is continuous approximation quality

The scalar head of xVal outputs a **continuous float**. This is appropriate and powerful for scientific prediction, but creates a fundamental mismatch when evaluated on tasks requiring exact integer outputs.

---

## Slide 7 — Dataset 1: ToTTo (Table-to-Text Faithful Copying)

**Task:** Given a linearised Wikipedia table with highlighted cells, generate a faithful one-sentence description.

**Why this tests numeric faithfulness:**
- Tables contain sports statistics, historical dates, electoral percentages, financial figures
- The model must **copy exact numbers** from the table into the sentence — no arithmetic required
- Wrong numbers, extra hallucinated numbers, or missing numbers all count as failures

**How we linearise ToTTo:**

ToTTo tables are hierarchical (section headers, sub-headers, row headers, cell values). We flatten them into a structured token sequence:

```
[TITLE] List of Governors of South Carolina [TITLE]
[COL] Governor | Political Party | Term Start | Term End [COL]
[ROW] Daniel Henry Chamberlain | Republican | 1874 | 1876 [ROW]
[SEP]
<target>: Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
```

**Key design choices:**
- `[TITLE]`, `[COL]`, `[ROW]`, `[SEP]` are special tokens added to the vocabulary of all 4 models
- For **xVal and FoNE**: numbers in both the table (input) and the target (output) are replaced with `[NUM]`, and their actual values are stored in a separate tensor
- For **Digit**: numbers are split digit-by-digit throughout the entire sequence
- Maximum sequence length: 256 tokens (some Digit examples exceed this due to digit expansion)

**Final evaluation scale:** 7,700 dev examples total; final benchmark on **200 examples** (179 of which contain numbers).

---

## Slide 8 — Dataset 2: BigBench Arithmetic (Exact Computation)

**Task:** Given a natural language arithmetic question, generate the correct integer answer.

**Why this tests numeric computation:**
- No number is present in context to copy — the model must **compute** the result
- Four operations: addition, subtraction, multiplication, division
- Numbers range from single-digit to approximately 4 digits
- The answer is always a single integer (or simple decimal)

**Example:**
```
Input:  "What is 87 plus 684?"
Target: "771"

Input:  "What is 390 divided by 5?"
Target: "78"
```

**Evaluation metrics:**
- **Exact Match** — generated string equals target exactly
- **Near Match (±1%)** — generated number is within 1% of target (catches rounding artefacts)

**Scale:** 4,000 train / 500 test; evaluated on all **500 test examples**.

**Why BigBench for this study?** It is clean, synthetically generated, balanced across operations, and isolates pure arithmetic from reading comprehension. There are no confounds from linguistic understanding — if the model gets it wrong, the number representation is the reason.

---

## Slide 9 — Results: Complete 4×2 Matrix

### ToTTo — Faithful Number Copying *(n=200 dev examples, 179 contain numbers)*

| Model | BLEU ↑ | Num-Exact-Match ↑ | Hallucination Rate ↓ | Count-Match ↑ |
|---|---|---|---|---|
| **BPE** | **33.41** | **90.6%** | **10.6%** | **70.0%** |
| Digit | 25.88 | 72.5% | 26.3% | 61.6% |
| FoNE | 18.41 | 72.2% | 53.5% | 26.0% |
| xVal | 15.12 | 21.3% | **82.0%** | 33.0% |

*Num-EM = fraction of reference numbers that appear correctly in the generated text.*
*Hallucination Rate = fraction of generated numbers NOT present in the reference.*

### BigBench Arithmetic — Exact Computation *(n=500 test examples)*

| Model | Exact Accuracy ↑ | Near Accuracy (±1%) ↑ |
|---|---|---|
| **FoNE** | **44.6%** | **51.6%** |
| Digit | 10.0% | 17.2% |
| BPE | 2.0% | 5.6% |
| **xVal** | **0.2%** | 2.2% |

### The Core Empirical Finding

**Rankings completely invert between the two tasks:**
- Best at copying (BPE) = near-worst at computation
- Best at computation (FoNE) = second-worst at copying
- xVal, designed for continuous prediction, is last on both exact-match tasks

---

## Slide 10 — Why These Results? Mechanistic Explanations

### BPE: Copy Expert, Computation Illiterate

BPE treats `"1874"` as an opaque vocabulary item — exactly like `"London"`. Self-attention layers learn **copy attention**: direct alignment of the source token to the output position.

This is efficient and reliable for number copying. But BPE has no mechanism to compute `87 + 684`. The opaque tokens `"87"` and `"684"` have no algebraic relationship in BPE's representation.

**Result:** 90.6% Num-EM on ToTTo, **2.0%** Exact Acc on BigBench.

---

### FoNE: Computation Expert, Text Generation Disruptor

Fourier embedding + MSE regression = strong inductive bias for arithmetic. The model must represent the actual numeric value at every `[NUM]` position to minimise the regression loss.

But when generating **free-form natural language** (ToTTo), the scalar regression head interferes with the language model's probability distribution. On examples with multiple `[NUM]` tokens in close proximity (e.g., a table row with 5 statistics), FoNE generates **runaway concatenated numbers** (`2016200220022002…`) or **endlessly repeated values**. Hallucination rate: **53.5%**, BLEU drops to 18.41.

---

### Digit: The Careful Middle Ground

Individual-digit attention enables partial arithmetic learning. But compounding autoregressive errors are the limiting factor: generating the 4-digit number `"1874"` requires 4 consecutive correct next-token predictions. A single error (predicting `"7"` instead of `"8"`) produces `"1774"` — the entire number is wrong.

Additionally, digit-split sequences are 2–3× longer than BPE sequences, making attention alignment harder across longer contexts.

---

### xVal: A Domain-Mismatched Model

xVal's two core problems on our tasks:

**On BigBench (arithmetic):** The scalar head outputs a continuous float (e.g., `151.6` for target `207`). The exact-match metric compares strings — `"151.6" ≠ "207"`. A model that outputs continuous values is **structurally incapable** of achieving high exact integer match. The near-match accuracy (2.2%) is more informative — xVal is directionally correct but not discrete.

**On ToTTo (text generation):** The scaled embedding `x · e_[NUM]` disrupts the language model's token probability distribution. After emitting one `[NUM]` token, the model's residual stream is biased towards generating more numbers. The result: **hallucination loops** of 10–20 consecutive float numbers. Hallucination rate: **82.0%**, BLEU: 15.12.

> **Important nuance:** xVal is not a "bad" model. It was not designed for NLP text generation or integer computation benchmarks. Its poor performance here is a **design-domain mismatch**, not a fundamental failure of continuous number embeddings.

---

## Slide 11 — The Structural Trade-Off

```
               BPE            Digit          FoNE           xVal
              ──────         ──────         ──────         ──────
Representation Discrete       Discrete       Continuous     Continuous
               Subword        Per-digit      Fourier        Value-scaled

Copy attention ✅ Direct      ⚠️ Fragmented  ❌ Disrupted   ❌❌ Broken
Arithmetic     ❌ No signal   ⚠️ Compounds   ✅ Supervised  ✅ But float

BLEU           33.41          25.88          18.41          15.12
Num-EM         90.6%          72.5%          72.2%          21.3%
BigBench       2.0%           10.0%          44.6%          0.2%
```

**The trade-off is structural and follows from the representation choices:**
- Discrete representations preserve the identity of number tokens → ideal for attention-based copying
- Continuous representations enable arithmetic supervision → ideal for mathematical computation
- No current single approach perfectly serves both without architectural modification

---

## Slide 12 — Proposed Novel Approaches

### What Recent Literature Tells Us

1. **Mechanistic interpretability** (Nanda et al. 2023; Grokking studies): Pretrained LLMs naturally develop Fourier-like internal representations for arithmetic. MLP layers encode magnitude via low-frequency features; attention layers encode modular arithmetic via high-frequency features. This suggests continuous embeddings can *emerge* without explicit supervision — but the question is whether they survive the language modelling objective on mixed text.

2. **Abacus Embeddings** (McLeish et al., NeurIPS 2024): Adding explicit **digit-position embeddings** (units=0, tens=1, hundreds=2...) achieves ~99% accuracy on 100-digit addition when trained only on 20-digit problems — strong length generalisation without changing the core architecture.

3. **AG-BPE** (2024): Semantic-aware BPE tokenisation that uses guidance to prevent fragmentation of semantically coherent units (including numbers).

---

### Proposal A: Gated Numeric Router (GNR)

**Core idea:** Add a lightweight **learnable gate** inside each transformer layer that routes `[NUM]` tokens through a specialised numeric sub-network:

```
Input token t
      ↓
  Is t == [NUM]?
  ┌──────┴───────┐
  ↓ gate=1       ↓ gate=0
Numeric path   Language path
(FoNE-style    (standard attention
 regression)    + MLP)
  └──────┬───────┘
         ↓
  Combined hidden state
```

```python
gate = torch.sigmoid(self.gate_proj(hidden_state))   # (0, 1) per token
numeric_out  = self.numeric_subnetwork(hidden_state)  # FoNE-style
language_out = self.language_subnetwork(hidden_state) # standard
output = gate * numeric_out + (1 - gate) * language_out
```

**Why this works:** The gate is trained end-to-end. On copy-heavy tasks (ToTTo), the gate learns to use the language path for number tokens, preserving their subword identity. On compute-heavy tasks (BigBench), the gate routes to the numeric path for regression-style computation. The gate provides a **soft, learned switch** rather than a hard architectural choice.

---

### Proposal B: Domain-Adaptive Numeric Tokenizer (DANT)

**Core idea:** At tokenization time, detect the task domain and apply the domain-appropriate strategy:

- **Copying domain** (table-to-text, summarisation): Retain numbers as complete **BPE subword tokens** for faithful copying
- **Computation domain** (arithmetic, scientific prediction): Replace numbers with `[NUM]` + Fourier embedding for arithmetic supervision

A **task prefix token** added to the sequence — `[COPY-TASK]` or `[COMPUTE-TASK]` — conditions the model's behaviour downstream without changing its architecture.

**Why this is novel:** Current approaches assume a fixed tokenisation strategy across all tasks. DANT makes number representation a **learnable, task-conditioned choice** — essentially meta-learning over tokenisation strategies.

---

### Proposal C: Abacus-Enhanced Digit Tokenization

**Core idea:** Extend the current Digit model with **positional embeddings relative to the start of each number** (inspired by McLeish et al., NeurIPS 2024):

```
Number:    1    8    7    4
Position:  3    2    1    0    ← place index (0=units, 1=tens, 2=hundreds...)
```

```python
# During embedding: add digit-place embedding at each digit token position
digit_place_emb = nn.Embedding(MAX_DIGITS, n_embd)
token_embeddings[digit_positions] += digit_place_emb(place_indices)
```

This preserves the **discreteness advantage** of Digit tokenisation (good for copying) while adding **structural arithmetic bias** (each digit knows its positional role in the number). Based on Abacus results (NeurIPS 2024), this should substantially improve BigBench accuracy from the current 10.0%.

---

## Slide 13 — Further Models to Study

Based on the research overview, the following models are **practically replicable** and ideal for extending this study:

| Model | Why Include | Reproducibility | Expected Strength |
|---|---|---|---|
| **Abacus Embeddings** | Digit-positional; strong arithmetic length generalisation; public code (mcleish7/arithmetic, NeurIPS 2024) | ✅ Public code + training scripts | BigBench arithmetic; length generalisation |
| **DICE** | Deterministic distance-preserving embeddings; cosine similarity reflects number-line distance; no training required | ✅ Trivial to implement (deterministic) | Copy-heavy tasks; may improve Num-EM |
| **NumGPT** | Mantissa-exponent prototype embeddings; numeral-aware loss during pretraining; covers scientific notation | ⚠️ Re-implement from paper description | Mixed tasks; large/small number ranges |
| **Vanilla char-digit** | Simplest baseline — each of 0–9 as its own character token; no BPE merges at all | ✅ Trivial to implement | Clean ablation to isolate Metaspace effect in our Digit model |

**Not recommended at this stage:**
- **NIF (Neural Isomorphic Fields):** Very recent (2026 preprint), no public code or model available
- **MWNE (Multi-Wavelet):** Designed for time-series forecasting; will likely fail on NLP tasks for the same structural reasons as xVal

---

## Slide 14 — Further Datasets to Study

### Computation Track — Tasks That Require Arithmetic

| Dataset | Why Include | Status |
|---|---|---|
| **SVAMP** | Tests arithmetic robustness against structural rephrasing of word problems; exposes shallow heuristics | ✅ Public |
| **DROP** | Discrete reasoning (addition, counting, sorting, comparison) embedded in reading passages | ✅ Public |
| **GSM8K** | Grade-school multi-step word problems; gold standard for arithmetic reasoning in NLP research | ✅ Public |
| **Abacus Addition Sets** | N-digit addition for length generalisation: train on 20-digit, test on 100-digit | ✅ Public |

### Faithfulness Track — Tasks That Require Faithful Copying

| Dataset | Why Include | Status |
|---|---|---|
| **LogicNLG** | Table-to-text requiring **logical inference**, not just copying; tests compute + copy combined | ✅ Public |
| **NumericNLG** | Specifically designed to test numeric accuracy in data-to-text generation | ✅ Public |
| **SummEdits** | Multi-domain inconsistency detection benchmark with explicit "number swap" error category | ✅ Public |
| **DART** | Open-domain data-to-text across diverse structured inputs | ✅ Public |

### The Gap: Tasks Requiring Both Computation and Copying

**ToTTo** tests *only* copying — numbers appear verbatim in the source table.  
**BigBench Arithmetic** tests *only* pure computation — no text is generated.

Real-world tasks require **both simultaneously**:
- A clinical discharge summariser must *copy* lab values (sodium: 142 mEq/L) *and compute* drug dose adjustments (body weight × 2.5 mg/kg)
- A financial report generator must *copy* individual figures *and compute* totals, differences, and percentages

**LogicNLG** and **DROP** are the most realistic benchmarks in this combined regime and should be the next evaluation targets.

---

## Slide 15 — Conclusions and Research Agenda

### Confirmed Findings

1. **Numeric computation ≠ Numeric generation faithfulness.** Rankings fully invert between BigBench (computation) and ToTTo (copying). This is the core hypothesis — **confirmed**.

2. **BPE's copying advantage is structural.** Opaque subword tokens are ideal for attention-based copy operations. Discrete representations will always have a natural advantage on faithful text generation tasks.

3. **FoNE's arithmetic advantage is structural.** Fourier regression supervision creates a direct inductive bias for magnitude and periodicity — the Fourier basis and MSE loss align perfectly with the requirements of arithmetic computation.

4. **xVal is a domain-mismatched model in our evaluation.** It was not designed for NLP text generation or integer arithmetic evaluation. Its poor performance reflects a mismatch between its design domain (continuous scientific prediction) and our evaluation tasks (exact-match integer arithmetic and NLP text generation).

5. **Digit tokenization is the best overall trade-off** among our 4 models — moderate performance on both tasks — and can be substantially improved with Abacus-style digit-place positional embeddings.

### Research Agenda

```
Phase 1 (✅ Done):  BPE | Digit | FoNE | xVal on ToTTo + BigBench (4×2 matrix)
Phase 2:            Add Abacus + DICE + Vanilla digit tokenisation
Phase 3:            Extend to DROP + LogicNLG (compute + copy combined tasks)
Phase 4:            Implement Gated Numeric Router (GNR); test DANT
Phase 5:            Ablation on decoding strategy (greedy vs. beam vs. sampling)
```

### The Core Takeaway for Future Work

> A single universal number representation does not exist. The right encoding is task-dependent.
> The most promising future direction is an architecture that can **choose** its number representation
> based on context — discrete for faithful text generation, continuous for arithmetic precision.

---

*Presentation prepared for internal research meeting.*
*All 4 models trained from scratch on identical GPT-2 architecture (~85M parameters) for fair comparison.*
*Results: ToTTo n=200, BigBench n=500.*
