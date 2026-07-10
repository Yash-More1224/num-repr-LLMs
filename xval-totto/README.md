# xval-totto — Numeric Generation Faithfulness Experiment

## Project structure

```
xval-totto/
├── requirements.txt
├── data/
│   ├── preprocess.py       ← Step 1: linearize ToTTo JSONL
│   ├── dataset.py          ← BPE dataset class
│   └── dataset_xval.py     ← xVal dataset class
├── tokenizer/
│   └── train_tokenizer.py  ← Step 2: train BPE tokenizer
├── model/                  ← (next phase)
├── train/                  ← (next phase)
└── eval/                   ← (next phase)
```

## Setup (run on cluster)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Move totto_data/ to the cluster (rsync or scp from local machine)
#    totto_data/ should sit one level above xval-totto/
#    i.e., the default path is ../totto_data/

# 3. cd into the project
cd xval-totto
```

## Phase 1: Data & Tokenizer

Run these in order from inside `xval-totto/`:

```bash
# Step 1 — Linearize the dataset (~5 min for 120K examples)
python data/preprocess.py \
    --data_dir ../totto_data \
    --output_dir processed

# Expected output files:
#   processed/train.jsonl          (~120K lines)
#   processed/dev.jsonl            (~7.7K lines)
#   processed/tokenizer_corpus.txt (~120K lines, same as train.jsonl linearized)

# Step 2 — Train the BPE tokenizer (~2 min)
python tokenizer/train_tokenizer.py \
    --corpus processed/tokenizer_corpus.txt \
    --output_dir tokenizer/totto_bpe \
    --vocab_size 16000

# Expected output files:
#   tokenizer/totto_bpe/tokenizer.json
#   tokenizer/totto_bpe/tokenizer_config.json

# Step 3 — Verify BPE dataset (optional sanity check)
python data/dataset.py

# Step 4 — Verify xVal dataset (optional sanity check, runs unit tests)
python data/dataset_xval.py
```

## Key design decisions

- **Both models share the exact same tokenizer** (trained in Step 2).
- **xVal replaces numbers with [NUM] before tokenization** — the regex runs
  in `dataset_xval.py` at load time, not in preprocess.py.
- **Loss is computed only on target tokens** (after [SEP]) in both models.
- **Numbers are log-normalized** (`sign * log1p(|x|)`) before storing as
  float values, so the regression head operates in a stable range.
