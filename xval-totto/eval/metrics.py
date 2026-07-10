"""
eval/metrics.py
===============
Shared numeric evaluation metrics used by both eval_bpe.py and eval_xval.py.

Metrics computed per example, then aggregated over the dev set:

1. BLEU          — sacrebleu corpus BLEU on full generated sentences
2. NUM-EM        — Numeric Exact Match: fraction of reference numbers
                   that appear exactly in the generated sentence
3. NUM-MAE       — Mean Absolute Error of paired numbers (reference vs generated)
4. NUM-MAPE      — Mean Absolute Percentage Error (%, relative to reference)
5. COUNT-MATCH   — fraction of examples where #generated_numbers == #reference_numbers
6. HALLUCINATION — fraction of generated numbers NOT in the reference set

For xVal only (computed in eval_xval.py):
7. SCALAR-MSE    — MSE of the scalar head's log-normalized predictions vs targets

Number Alignment Strategy:
  We pair numbers positionally (first generated ↔ first reference, etc.).
  If counts differ, we evaluate only the min(n_gen, n_ref) pairs and
  penalize via COUNT-MATCH.
"""

import re
import math
from typing import List, Tuple


# ── Number extraction ─────────────────────────────────────────────────────────

NUMBER_RE = re.compile(
    r"(?<!\w)"                    # not preceded by a word character
    r"(-?"                        # optional leading minus
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"  # integer with optional commas, or plain integer
    r"(?:\.\d+)?)"                # optional decimal part
    r"(?!\w)"                     # not followed by a word character
)


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric values from a text string.

    Handles:
      - Plain integers:  "127"      → [127.0]
      - Comma-formatted: "1,234"    → [1234.0]
      - Negatives:       "-5"       → [-5.0]
      - Decimals:        "3.14"     → [3.14]
      - Mixed:           "1,234.56" → [1234.56]

    Does NOT match:
      - Years inside words: "2000s" (because of the trailing 's')
      - Ordinals with suffix attached: "78th" ('th' is attached)

    Returns an empty list if no numbers are found.
    """
    values = []
    for m in NUMBER_RE.finditer(text):
        cleaned = m.group(1).replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            pass
    return values


# ── Per-example metric computation ────────────────────────────────────────────

def numeric_exact_match(ref_nums: List[float], gen_nums: List[float],
                        tol: float = 0.01) -> float:
    """
    Fraction of reference numbers that appear exactly (within tolerance tol)
    in the generated number list.

    tol=0.01 means numbers must be within 1% of each other.
    If reference has no numbers, returns 1.0 (vacuously correct).

    Example:
      ref_nums = [2018.0, 127.0]
      gen_nums = [2018.0, 126.0]
      → 1 out of 2 matched → 0.5
    """
    if not ref_nums:
        return 1.0

    matched = 0
    gen_remaining = list(gen_nums)   # we consume gen nums one by one

    for r in ref_nums:
        for i, g in enumerate(gen_remaining):
            # Relative tolerance for non-zero refs, absolute for zero
            if r == 0.0:
                if abs(g) <= tol:
                    matched += 1
                    gen_remaining.pop(i)
                    break
            else:
                if abs(g - r) / abs(r) <= tol:
                    matched += 1
                    gen_remaining.pop(i)
                    break

    return matched / len(ref_nums)


def numeric_mae(ref_nums: List[float], gen_nums: List[float]) -> float:
    """
    Mean Absolute Error of positionally-paired numbers.

    If lists have different lengths, evaluate min(n_ref, n_gen) pairs.
    Returns None if both lists are empty.

    Example:
      ref_nums = [2018.0, 127.0]
      gen_nums = [2018.0, 130.0]
      → MAE = mean(|2018-2018|, |127-130|) = mean(0, 3) = 1.5
    """
    if not ref_nums and not gen_nums:
        return None
    if not ref_nums or not gen_nums:
        # One is empty: use ref magnitudes as "absolute error" for unmatched
        return float(sum(abs(x) for x in (ref_nums or gen_nums)) /
                     len(ref_nums or gen_nums))

    n = min(len(ref_nums), len(gen_nums))
    mae = sum(abs(ref_nums[i] - gen_nums[i]) for i in range(n)) / n
    return mae


def numeric_mape(ref_nums: List[float], gen_nums: List[float]) -> float:
    """
    Mean Absolute Percentage Error (%) of positionally-paired numbers.

    Skips pairs where ref_num == 0 (undefined MAPE).
    Returns None if no valid pairs.
    """
    if not ref_nums or not gen_nums:
        return None

    n = min(len(ref_nums), len(gen_nums))
    errors = []
    for i in range(n):
        if ref_nums[i] != 0:
            errors.append(abs(ref_nums[i] - gen_nums[i]) / abs(ref_nums[i]) * 100.0)

    return sum(errors) / len(errors) if errors else None


def hallucination_rate(ref_nums: List[float], gen_nums: List[float],
                       tol: float = 0.01) -> float:
    """
    Fraction of GENERATED numbers that do NOT match any reference number.

    A generated number is "hallucinated" if it has no reference counterpart
    within relative tolerance tol.

    If no numbers were generated, returns 0.0 (no hallucinations).

    Example:
      ref_nums = [2018.0]
      gen_nums = [2018.0, 500.0]  ← 500 is not in reference
      → hallucination_rate = 1/2 = 0.5
    """
    if not gen_nums:
        return 0.0

    hallucinated = 0
    ref_remaining = list(ref_nums)

    for g in gen_nums:
        found = False
        for i, r in enumerate(ref_remaining):
            if r == 0.0:
                if abs(g) <= tol:
                    ref_remaining.pop(i)
                    found = True
                    break
            else:
                if abs(g - r) / abs(r) <= tol:
                    ref_remaining.pop(i)
                    found = True
                    break
        if not found:
            hallucinated += 1

    return hallucinated / len(gen_nums)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(results: List[dict]) -> dict:
    """
    Aggregate per-example metric dicts into a single summary dict.

    Each result dict should have:
      {
        "num_em":     float,   # Numeric Exact Match (per example)
        "num_mae":    float or None,
        "num_mape":   float or None,
        "halluc":     float,   # Hallucination rate
        "count_match": bool,   # #gen_nums == #ref_nums
        "bleu_ref":   str,     # reference sentence (for sacrebleu)
        "bleu_hyp":   str,     # generated sentence (for sacrebleu)
        "n_ref_nums": int,     # number of reference numbers
      }

    Returns a dict with macro-averaged metrics.
    """
    import sacrebleu

    n = len(results)
    if n == 0:
        return {}

    # Filter to only examples that had at least one number in the reference
    # for MAE/MAPE/EM (to avoid 1.0 vacuous matches distorting the stats)
    numeric_results = [r for r in results if r["n_ref_nums"] > 0]
    n_numeric = len(numeric_results)

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else float("nan")

    # BLEU over the full dev set
    refs  = [[r["bleu_ref"]  for r in results]]
    hyps  =  [r["bleu_hyp"]  for r in results]
    bleu  = sacrebleu.corpus_bleu(hyps, refs).score

    summary = {
        "n_examples":          n,
        "n_with_numbers":      n_numeric,
        "bleu":                round(bleu, 2),
        "num_em":              round(safe_mean([r["num_em"]  for r in numeric_results]), 4),
        "halluc_rate":         round(safe_mean([r["halluc"]  for r in results]), 4),
        "count_match":         round(sum(r["count_match"] for r in results) / n, 4),
        "num_mae":             round(safe_mean([r["num_mae"]  for r in numeric_results]), 4),
        "num_mape_pct":        round(safe_mean([r["num_mape"] for r in numeric_results]), 2),
    }
    return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing extract_numbers...")
    tests = [
        ("The Bears selected Roquan Smith in 2018.", [2018.0]),
        ("Thomas burned 281,893 acres.", [281893.0]),
        ("Elagabalus (204 - 11 March 222) was emperor.", [204.0, 11.0, 222.0]),
        ("No numbers here.", []),
        ("Born in the 2000s.", []),       # "2000s" should NOT match
    ]
    for text, expected in tests:
        got = extract_numbers(text)
        status = "✓" if got == expected else "✗"
        print(f"  {status} '{text[:50]}' → {got}  (expected {expected})")

    print("\nTesting numeric_exact_match...")
    assert numeric_exact_match([2018.0, 127.0], [2018.0, 127.0]) == 1.0
    assert numeric_exact_match([2018.0, 127.0], [2018.0, 130.0]) == 0.5
    assert numeric_exact_match([], []) == 1.0
    assert numeric_exact_match([100.0], []) == 0.0

    print("Testing hallucination_rate...")
    assert hallucination_rate([2018.0], [2018.0, 500.0]) == 0.5
    assert hallucination_rate([2018.0], [2018.0]) == 0.0
    assert hallucination_rate([2018.0], []) == 0.0

    print("\n✓ metrics.py OK")
