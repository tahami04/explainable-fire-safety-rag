"""Does the faithfulness check reject *false* claims? (offline, no API calls)

Takes every answer sentence from the latest full run that the verifier accepted,
makes deliberately false variants of it, and re-runs the same verification against
the same citations:

* number swap  – every number n becomes 2n (or 1 if n is 0): "11m" → "22m"
* negation     – the first modal/copula gets "not": "should be" → "should not be"

A good verifier keeps accepting the originals and rejects the variants. The
false-acceptance rate is the share of variants still labelled Supported (strict) or
Supported/Partially supported (lenient).

Usage:
    python eval/perturbation_check.py [--results eval/results/full.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firerag.config import get_settings  # noqa: E402
from firerag.models import Chunk  # noqa: E402
from firerag.verify.faithfulness import (  # noqa: E402
    check_sentence,
    default_scorer,
    premise_prefix_for,
    table_fallback,
)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_MODAL = re.compile(r"\b(should|must|may|can|will|is|are|does|do)\b(?! not)")


def swap_numbers(s: str) -> str | None:
    """Double every number; None if the sentence has no numbers."""
    if not _NUM.search(s):
        return None

    def dbl(m: re.Match[str]) -> str:
        v = float(m.group(0))
        out = 1 if v == 0 else v * 2
        return str(int(out)) if out == int(out) else str(out)

    return _NUM.sub(dbl, s)


def negate(s: str) -> str | None:
    """Insert 'not' after the first modal/copula; None if there is none."""
    return _MODAL.sub(lambda m: f"{m.group(1)} not", s, count=1) if _MODAL.search(s) else None


def main() -> None:
    """Run the perturbation check and print a summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "eval" / "results" / "full.json"))
    args = ap.parse_args()
    s = get_settings()
    rows = json.loads(Path(args.results).read_text())["rows"]
    chunks = {
        c.chunk_id: c
        for c in (Chunk.model_validate_json(x) for x in (s.index_dir / "chunks.jsonl").read_text().splitlines())
    }
    ev = {k: c.text for k, c in chunks.items()}
    pre = {k: premise_prefix_for(c.volume, c.context_header(), c.kind, c.paragraph_ref) for k, c in chunks.items()}
    scorer = default_scorer()

    def verify(sentence: str, cites: list[str], quotes: list[dict]) -> str:
        f = check_sentence(sentence, cites, ev, scorer, s.nli_supported, s.nli_partial, premise_prefix=pre)
        tables = {c: ev[c] for c in cites if chunks[c].kind == "table"}
        return table_fallback(f, sentence, tables, all(q["verbatim"] for q in quotes), bool(quotes)).label

    results: dict[str, list[str]] = {"original": [], "number swap": [], "negation": []}
    examples = []
    for row in rows:
        for a in row["answer"]:
            if a["faithfulness"] == "Unsupported":
                continue
            results["original"].append(verify(a["sentence"], a["citations"], a["quotes"]))
            for kind, fn in (("number swap", swap_numbers), ("negation", negate)):
                variant = fn(a["sentence"])
                if variant is None:
                    continue
                label = verify(variant, a["citations"], a["quotes"])
                results[kind].append(label)
                if label == "Supported" and len(examples) < 5:
                    examples.append({"kind": kind, "variant": variant, "label": label})

    summary = {}
    for kind, labels in results.items():
        n = len(labels)
        summary[kind] = {
            "n": n,
            "labelled_supported": round(sum(lab == "Supported" for lab in labels) / n, 3) if n else None,
            "labelled_supported_or_partial": round(sum(lab != "Unsupported" for lab in labels) / n, 3) if n else None,
        }
    out = {"source": Path(args.results).name, "summary": summary, "false_acceptance_examples": examples}
    (ROOT / "eval" / "results" / "perturbation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
