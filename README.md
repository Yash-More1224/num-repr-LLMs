# Benchmarking Numerical Representation Strategies in Autoregressive Language Models

This repository contains a unified framework for benchmarking and comparing **4 numerical representation strategies** across **2 distinct benchmark tasks** in Transformer-based Autoregressive Language Models (GPT-2 backbone), forming a complete **8-experiment evaluation matrix**.

---

## Benchmark Overview

### 1. Benchmark Datasets (2 Tasks)
* **ToTTo (Table-to-Text Generation):** Natural language generation from linearized tables (~120K train, ~7.7K dev). Evaluates numerical generation accuracy alongside natural language text.
* **BigBench Arithmetic (Mathematical Reasoning):** Multi-digit arithmetic reasoning (addition, subtraction, multiplication) (4,000 train, 500 dev, 500 test). Evaluates raw numerical computation capability.

### 2. Numerical Representation Models (4 Variants)
* **BPE (Subword Tokenization):** Standard GPT-2 autoregressive language model with subword tokenization.
* **xVal (Continuous Scalar Regression):** Maps numbers to a single `[NUM]` token with continuous log-normalized scalar embedding injection ($\mathbf{e}_{\text{[NUM]}} + \mathbf{w} \cdot x$) and joint Cross-Entropy + MSE regression loss.
* **FoNE (Fourier Number Embeddings):** Maps numbers to a single `[NUM]` token with continuous multi-period Fourier sinusoidal feature embeddings ($\Phi(x)$) and joint Cross-Entropy + multi-task Fourier digit classification loss. Decodes all digits in parallel via inverse Fourier transformation.
* **Digit Tokenization:** Pre-splits numbers character-by-character into space-separated single-digit tokens (`"1874"` $\rightarrow$ `"1 8 7 4"`). Autoregressive token-by-token digit generation.

---

## Repository Structure

```
numeric-eval/ (xval-totto/)
├── configs/
│   ├── train_config.yaml          # Hyperparameters for ToTTo dataset
│   ├── bigbench_train_config.yaml # Hyperparameters for BigBench dataset
│   └── model_config.yaml          # Architecture hyperparameters (GPT-2 backbone)
├── data/
│   ├── preprocess.py              # Linearize ToTTo JSONL dataset
│   ├── dataset.py                 # ToTTo BPE dataset class
│   ├── dataset_xval.py            # ToTTo xVal dataset class
│   ├── dataset_fone.py            # ToTTo FoNE dataset class
│   ├── dataset_digit.py           # ToTTo Digit dataset class
│   └── dataset_bigbench.py        # BigBench dataset class (supports all 4 models)
├── tokenizer/
│   ├── train_tokenizer.py         # Trains BPE tokenizer
│   └── train_digit_tokenizer.py   # Creates custom single-digit tokenizer
├── model/
│   ├── gpt2.py                    # GPT2BPE architecture
│   ├── gpt2_xval.py               # GPT2XVal architecture
│   ├── gpt2_fone.py               # GPT2FoNE architecture
│   └── gpt2_digit.py              # GPT2Digit architecture
├── train/
│   ├── trainer.py                 # Core trainer loop (DDP, AMP, Early Stopping)
│   ├── train_bpe.py               # ToTTo BPE training entrypoint
│   ├── train_xval.py              # ToTTo xVal training entrypoint
│   ├── train_fone.py              # ToTTo FoNE training entrypoint
│   ├── train_digit.py             # ToTTo Digit training entrypoint
│   └── train_bigbench.py          # BigBench training entrypoint (all 4 models)
└── eval/
    ├── metrics.py                 # Exact Match, MAE, MAPE, Hallucination Rate
    ├── eval_bpe.py                # ToTTo BPE evaluation script
    ├── eval_xval.py               # ToTTo xVal evaluation script
    ├── eval_fone.py               # ToTTo FoNE evaluation script
    ├── eval_digit.py              # ToTTo Digit evaluation script
    └── eval_bigbench.py           # BigBench evaluation script (all 4 models)
```

---

## Usage Guide

### 1. Requirements & Installation

```bash
pip install -r requirements.txt
```

### 2. ToTTo Dataset Pipeline

```bash
# Step 1 — Linearize ToTTo dataset
python data/preprocess.py --data_dir ../totto_data --output_dir processed

# Step 2 — Train Tokenizers
python tokenizer/train_tokenizer.py --corpus processed/tokenizer_corpus.txt --output_dir tokenizer/totto_bpe --vocab_size 16000
python tokenizer/train_digit_tokenizer.py

# Step 3 — Distributed Multi-GPU Training (e.g. FoNE or Digit on ToTTo)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --nproc_per_node=4 train/train_fone.py \
    --config configs/train_config.yaml \
    --model_config configs/model_config.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --nproc_per_node=4 train/train_digit.py \
    --config configs/train_config.yaml \
    --model_config configs/model_config.yaml

# Step 4 — Evaluation on ToTTo
python eval/eval_fone.py \
    --checkpoint checkpoints/fone/fone_best_epoch*.pt \
    --config configs/model_config.yaml \
    --train_config configs/train_config.yaml \
    --output results/fone_totto.json
```

### 3. BigBench Arithmetic Pipeline

```bash
# Train any model on BigBench (model_type = bpe | xval | fone | digit)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --nproc_per_node=4 train/train_bigbench.py \
    --model_type fone \
    --config configs/bigbench_train_config.yaml \
    --model_config configs/model_config.yaml

# Evaluate on BigBench Test Set
python eval/eval_bigbench.py \
    --model_type fone \
    --checkpoint checkpoints/fone_bigbench/fone_bigbench_best_*.pt \
    --config configs/model_config.yaml \
    --train_config configs/bigbench_train_config.yaml \
    --test_data ../data/bigbench_arithmetic/test.jsonl \
    --output results/bigbench_fone.json
```

---

## Evaluation Metrics

All models are evaluated on identical metrics for unified comparison:
1. **Exact Match (EM):** Exact numerical equality ($x_{\text{pred}} == x_{\text{target}}$).
2. **Near Accuracy (±1% Tolerance):** Relative error $|x_{\text{pred}} - x_{\text{target}}| / |x_{\text{target}}| \le 0.01$.
3. **MAE / MAPE:** Mean Absolute Error & Mean Absolute Percentage Error.
4. **Hallucination Rate:** Frequency of non-ground-truth numerical generations.
