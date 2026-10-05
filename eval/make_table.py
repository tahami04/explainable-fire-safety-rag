"""Print the README results table from eval/results/*.json (numbers are copied, never typed).

Usage:
    python eval/make_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def main() -> None:
    """Print a markdown table of the latest retrieval-only and full runs."""
    ret = json.loads((RESULTS / "retrieval.json").read_text())["metrics"]
    full_path = RESULTS / "full.json"
    full = json.loads(full_path.read_text()) if full_path.exists() else None
    m = full["metrics"] if full else {}
    n_oos_abst = round(m.get("abstention_accuracy_oos", 0) * m.get("n_out_of_scope", 0)) if full else None
    n_ans = round(m.get("in_scope_answer_rate", 0) * m.get("n_in_scope", 0)) if full else None
    labels = m.get("faithfulness_labels", {})
    rows = [
        ("Retrieval hit@5 (in-scope)", _fmt(ret["retrieval_hit_at_5"]), f"{round(ret['retrieval_hit_at_5'] * 20)}/20"),
        ("Retrieval MRR@5", _fmt(ret["retrieval_mrr_at_5"], pct=False), ""),
        (
            "Abstention accuracy, out-of-scope (full pipeline)",
            _fmt(m.get("abstention_accuracy_oos")),
            f"{n_oos_abst}/5" if full else "",
        ),
        (
            "Abstention, out-of-scope (retrieval gate only)",
            _fmt(ret["abstention_accuracy_oos"]),
            f"{round(ret['abstention_accuracy_oos'] * 5)}/5",
        ),
        (
            "In-scope questions answered (not abstained)",
            _fmt(m.get("in_scope_answer_rate")),
            f"{n_ans}/20" if full else "",
        ),
        (
            "Faithfulness: sentences Supported (NLI)",
            _fmt(m.get("faithfulness_supported")),
            f"{labels.get('Supported', 0)}/{m.get('n_answer_sentences', 0)}" if full else "",
        ),
        ("Faithfulness: Supported or Partially supported", _fmt(m.get("faithfulness_supported_or_partial")), ""),
        ("Citation validity (cited ID was retrieved)", _fmt(m.get("citation_validity")), ""),
        ("Quotes found verbatim in cited passage", _fmt(m.get("quote_verbatim_rate")), ""),
        ("Citation precision (cited passage entails sentence, NLI)", _fmt(m.get("citation_precision_nli")), ""),
        ("Citation precision (cited = expected paragraph; strict)", _fmt(m.get("citation_precision_expected")), ""),
        ("Mean self-consistency (answered in-scope)", _fmt(m.get("mean_consistency_in_scope"), pct=False), ""),
        (
            "Latency per question, mean / median / max (full, N=5)",
            f"{m['latency_mean_s']}s / {m['latency_median_s']}s / {m['latency_max_s']}s" if full else "—",
            "",
        ),
        ("Latency per question, mean (retrieval-only)", f"{ret['latency_mean_s']}s", ""),
    ]
    print("| Metric | Result | Count |\n|---|---|---|")
    for name, val, cnt in rows:
        print(f"| {name} | {val} | {cnt} |")
    if full:
        print(f"\nModel: `{full['model']}` · run {full['run_at']} · {full['consistency_samples']} consistency samples")


if __name__ == "__main__":
    main()
