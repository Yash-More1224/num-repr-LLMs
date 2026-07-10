"""
eval/compare.py
===============
Load saved BPE and xVal result JSON files and produce a side-by-side
comparison report.

Also samples and prints examples where the two models diverge on numeric
accuracy, which is useful for qualitative analysis.

Usage:
  python eval/compare.py \\
      --bpe   results/bpe_results.json \\
      --xval  results/xval_results.json \\
      --output results/comparison_report.txt \\
      --n_examples 10    # number of qualitative examples to print
"""

import json
import argparse
from pathlib import Path


# ── Formatting helpers ────────────────────────────────────────────────────────

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"

def fmt(value, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (value != value):   # NaN check
        return "N/A"
    return f"{value:.{decimals}f}"

def delta(bpe_val: float, xval_val: float, higher_is_better: bool) -> str:
    """Return a colored arrow showing which model is better."""
    if bpe_val is None or xval_val is None:
        return ""
    diff = xval_val - bpe_val
    if higher_is_better:
        return f"  xVal {'↑' if diff > 0 else '↓'} by {abs(diff):.4f}"
    else:
        return f"  xVal {'↑' if diff > 0 else '↓'} by {abs(diff):.4f}  ({'better' if diff < 0 else 'worse'})"


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(bpe: dict, xval: dict, n_examples: int) -> str:
    bs = bpe["summary"]
    xs = xval["summary"]

    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 01 — BPE vs xVal GPT-2 Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Checkpoints:")
    lines.append(f"  BPE  : {bpe['checkpoint']}")
    lines.append(f"  xVal : {xval['checkpoint']}")
    lines.append("")

    lines.append("─" * 70)
    lines.append(f"{'Metric':<35} {'BPE':>10} {'xVal':>10} {'Winner':>10}")
    lines.append("─" * 70)

    def row(name, bpe_val, xval_val, higher_is_better, decimals=4):
        bv = fmt(bpe_val, decimals)
        xv = fmt(xval_val, decimals)
        if bpe_val is None or xval_val is None:
            winner = "N/A"
        elif higher_is_better:
            winner = "xVal ✓" if xval_val > bpe_val else ("BPE ✓" if bpe_val > xval_val else "tie")
        else:
            winner = "xVal ✓" if xval_val < bpe_val else ("BPE ✓" if bpe_val < xval_val else "tie")
        lines.append(f"  {name:<33} {bv:>10} {xv:>10} {winner:>10}")

    row("BLEU Score",                bs.get("bleu"),          xs.get("bleu"),          higher_is_better=True,  decimals=2)
    row("Numeric Exact Match",       bs.get("num_em"),        xs.get("num_em"),        higher_is_better=True)
    row("Hallucination Rate",        bs.get("halluc_rate"),   xs.get("halluc_rate"),   higher_is_better=False)
    row("Count Match Rate",          bs.get("count_match"),   xs.get("count_match"),   higher_is_better=True)
    row("Numeric MAE",               bs.get("num_mae"),       xs.get("num_mae"),       higher_is_better=False)
    row("Numeric MAPE (%)",          bs.get("num_mape_pct"),  xs.get("num_mape_pct"),  higher_is_better=False, decimals=2)

    lines.append("─" * 70)
    lines.append(f"  {'Scalar MSE (log-norm)':<33} {'N/A':>10} {fmt(xs.get('scalar_mse_lognorm'), 6):>10}   [xVal only]")
    lines.append("─" * 70)
    lines.append("")

    # ── Interpretation block ──────────────────────────────────────────────────
    lines.append("INTERPRETATION")
    lines.append("─" * 70)

    bpe_num_em  = bs.get("num_em", 0)
    xval_num_em = xs.get("num_em", 0)

    if xval_num_em > bpe_num_em:
        lines.append(f"✅ xVal has HIGHER Numeric Exact Match ({pct(xval_num_em)} vs {pct(bpe_num_em)}).")
        lines.append("   This supports the hypothesis that continuous number embeddings")
        lines.append("   improve numeric fidelity in generation.")
    elif bpe_num_em > xval_num_em:
        lines.append(f"⚠️  BPE has HIGHER Numeric Exact Match ({pct(bpe_num_em)} vs {pct(xval_num_em)}).")
        lines.append("   The xVal embedding did NOT improve numeric accuracy here.")
        lines.append("   Possible reasons: insufficient training, mse_lambda too high,")
        lines.append("   or the task is dominated by copy-from-input patterns.")
    else:
        lines.append("→ Numeric Exact Match is equal.")

    bpe_bleu  = bs.get("bleu", 0)
    xval_bleu = xs.get("bleu", 0)
    if abs(bpe_bleu - xval_bleu) < 1.0:
        lines.append(f"\n→ BLEU scores are similar ({bpe_bleu:.2f} vs {xval_bleu:.2f}).")
        lines.append("   The xVal encoding did not hurt text generation quality.")
    elif xval_bleu > bpe_bleu:
        lines.append(f"\n✅ xVal has HIGHER BLEU ({xval_bleu:.2f} vs {bpe_bleu:.2f}).")
    else:
        lines.append(f"\n⚠️  xVal has LOWER BLEU ({xval_bleu:.2f} vs {bpe_bleu:.2f}).")
        lines.append("   Text quality suffered — investigate whether CE loss is converging properly.")

    lines.append("")

    # ── Qualitative examples ──────────────────────────────────────────────────
    if n_examples > 0:
        lines.append("─" * 70)
        lines.append(f"QUALITATIVE EXAMPLES (examples with most divergent numeric accuracy)")
        lines.append("─" * 70)

        bpe_ex  = {e["example_id"]: e for e in bpe["examples"]}
        xval_ex = {e["example_id"]: e for e in xval["examples"]}

        # Find examples where the two models differ in numeric accuracy
        common_ids = set(bpe_ex.keys()) & set(xval_ex.keys())
        divergent = []
        for eid in common_ids:
            b = bpe_ex[eid]
            x = xval_ex[eid]
            if b["n_ref_nums"] > 0:
                em_diff = abs(b["num_em"] - x["num_em"])
                divergent.append((em_diff, eid))
        divergent.sort(reverse=True)

        for _, eid in divergent[:n_examples]:
            b = bpe_ex[eid]
            x = xval_ex[eid]
            lines.append(f"\nExample ID: {eid}")
            lines.append(f"  Reference : {b['reference']}")
            lines.append(f"  Ref nums  : {b['ref_nums']}")
            lines.append(f"  BPE gen   : {b['generated']}")
            lines.append(f"  BPE nums  : {b['gen_nums']}  (EM={b['num_em']:.2f})")
            lines.append(f"  xVal gen  : {x['generated']}")
            lines.append(f"  xVal nums : {x['gen_nums']}  (EM={x['num_em']:.2f})")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe",         required=True,  help="Path to bpe_results.json")
    parser.add_argument("--xval",        required=True,  help="Path to xval_results.json")
    parser.add_argument("--output",      default="results/comparison_report.txt")
    parser.add_argument("--n_examples",  type=int, default=10)
    args = parser.parse_args()

    with open(args.bpe) as f:
        bpe_data = json.load(f)
    with open(args.xval) as f:
        xval_data = json.load(f)

    report = generate_report(bpe_data, xval_data, args.n_examples)
    print(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
