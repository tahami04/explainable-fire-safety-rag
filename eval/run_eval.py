"""Run the evaluation set and write real results to eval/results/.

Usage:
    python eval/run_eval.py --mode retrieval   # no API key needed
    python eval/run_eval.py --mode full        # needs an LLM provider configured in .env
    Add --fresh to discard cached per-question results (they let a rate-limited run resume).
    python eval/run_eval.py --mode full --replay   # re-verify recorded generations, no API calls

Metric definitions (also written into the results file):

* hit@5 / MRR@5     – an expected chunk appears in the top 5 retrieved passages / reciprocal rank.
* answer rate       – share of in-scope questions answered (not abstained).
* faithfulness      – share of answer sentences labelled Supported by the NLI check
                      (strict), and Supported + Partially supported (lenient).
* citation validity – share of model citations that point to a retrieved passage.
* quote verbatim    – share of quotes found verbatim in the passage they cite.
* citation precision (NLI)      – share of valid citations whose passage alone entails
                                  (≥ partial threshold) the sentence citing it.
* citation precision (expected) – share of valid citations that are one of the question's
                                  expected chunks. Strict: other relevant paragraphs count as misses.
* abstention accuracy (OOS)     – share of out-of-scope questions where the system abstained.
* latency           – wall-clock seconds per question, end to end.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firerag.config import get_settings  # noqa: E402
from firerag.generate.providers import RecordingProvider, ReplayProvider  # noqa: E402
from firerag.pipeline import AskResult, Pipeline  # noqa: E402


def _matches(chunk_id: str, expected: list[str]) -> bool:
    """A retrieved chunk matches if it is an expected ID or a window/duplicate of one."""
    return any(chunk_id == e or chunk_id.startswith((f"{e}-w", f"{e}~")) for e in expected)


def _live_latencies(rows: list[dict]) -> list[float]:
    return [r["latency_s"] for r in rows if not r.get("replayed")]


def _pct(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _cache_path(mode: str, model: str | None) -> Path:
    return ROOT / "eval" / "results" / f".cache_{mode}_{(model or 'none').replace('/', '_')}.jsonl"


def evaluate(mode: str, n_samples: int | None, fresh: bool = False, replay: bool = False) -> dict:
    """Run every question and compute aggregate metrics."""
    s = get_settings()
    qs = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text())
    pipe = Pipeline(audit=False)
    pipe.raise_provider_errors = True
    run_mode = "full" if mode == "full" else "retrieval_only"
    if run_mode == "full" and pipe.provider is None and not replay:
        raise SystemExit("Full mode needs an LLM provider: set OPENAI_API_KEY (or LLM_PROVIDER=ollama) in .env")
    if run_mode == "full":
        # Every LLM call is recorded so the verification layer can be re-evaluated
        # later against identical generations (--replay), at no API cost.
        name = pipe.provider.name if pipe.provider else s.llm_provider
        model_name = pipe.provider.model if pipe.provider else s.openai_model
        rec_path = ROOT / "eval" / "results" / f"llm_calls_{model_name}.jsonl"
        if replay:
            pipe.provider = ReplayProvider(rec_path, name, model_name)
        else:
            pipe.provider = RecordingProvider(pipe.provider, rec_path)
    pipe.ask("warm-up: protected shafts", mode="retrieval_only")  # load models before timing

    # Per-question results are cached as they complete so a run interrupted by an API
    # rate limit can resume. The cache key includes model and prompt version.
    model = pipe.provider.model if (pipe.provider and run_mode == "full") else None
    cache_file = _cache_path(run_mode, model)
    cache: dict[str, dict] = {}
    if fresh and cache_file.exists():
        cache_file.unlink()
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            if row.get("prompt_version") == s.prompt_version:
                cache[row["id"]] = row
    rows = []
    for item in qs["in_scope"] + qs["out_of_scope"]:
        if item["id"] in cache:
            rows.append(cache[item["id"]])
            print(f"{item['id']}  (cached)")
            continue
        in_scope = "expected" in item
        t0 = time.perf_counter()
        try:
            r: AskResult = pipe.ask(item["question"], mode=run_mode, n_samples=n_samples)
        except Exception as exc:  # e.g. openai.RateLimitError
            raise SystemExit(
                f"\nStopped at {item['id']}: {type(exc).__name__}: {str(exc)[:300]}\n"
                f"{len(rows)} results are cached in {cache_file.relative_to(ROOT)}; re-run the same command to resume."
            ) from exc
        latency = time.perf_counter() - t0
        exp = item.get("expected", [])
        top5 = [e.chunk_id for e in r.evidence[:5]]
        rank = next((i for i, cid in enumerate(top5, 1) if _matches(cid, exp)), None)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "in_scope": in_scope,
                "expected": exp,
                "top5": top5,
                "hit_rank": rank,
                "abstained": r.abstained,
                "abstention_reason": r.abstention_reason,
                "retrieval_strength": r.confidence.retrieval_strength,
                "confidence": r.confidence.model_dump(),
                "consistency": r.consistency.model_dump() if r.consistency else None,
                "answer": [a.model_dump() for a in r.answer],
                "caveats": r.caveats,
                "latency_s": round(latency, 2),
                "latency_breakdown_ms": r.latency_ms,
                "prompt_version": r.prompt_version,
                "model": r.model,
                "replayed": replay,
            }
        )
        with cache_file.open("a") as fh:
            fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        print(
            f"{item['id']}  hit@5={'Y' if rank else 'n'}  abstained={r.abstained!s:5}  "
            f"conf={r.confidence.label:6}  {latency:5.1f}s  {item['question'][:60]}"
        )

    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    metrics: dict = {
        "n_in_scope": len(ins),
        "n_out_of_scope": len(oos),
        "retrieval_hit_at_5": _pct([1.0 if r["hit_rank"] else 0.0 for r in ins]),
        "retrieval_mrr_at_5": _pct([1.0 / r["hit_rank"] if r["hit_rank"] else 0.0 for r in ins]),
        "abstention_accuracy_oos": _pct([1.0 if r["abstained"] else 0.0 for r in oos]),
        "in_scope_answer_rate": _pct([0.0 if r["abstained"] else 1.0 for r in ins]),
        # Replayed rows have no LLM time, so latency is computed from live rows only.
        "latency_mean_s": round(statistics.mean(lat), 2) if (lat := _live_latencies(rows)) else None,
        "latency_median_s": round(statistics.median(lat), 2) if lat else None,
        "latency_max_s": round(max(lat), 2) if lat else None,
        "latency_n_live_questions": len(lat),
        "oos_retrieval_strength": {r["id"]: r["retrieval_strength"] for r in oos},
        "in_scope_retrieval_strength_min": min(r["retrieval_strength"] for r in ins),
        # How the pre-generation gate alone would trade off refusals against answers.
        # Reported for transparency; the shipped threshold is NOT tuned on this set.
        "gate_sensitivity": [
            {
                "threshold": t,
                "oos_refused": _pct([1.0 if r["retrieval_strength"] < t else 0.0 for r in oos]),
                "in_scope_passed": _pct([1.0 if r["retrieval_strength"] >= t else 0.0 for r in ins]),
            }
            for t in (0.05, 0.1, 0.2, 0.5, 0.9, 0.99)
        ],
    }
    if run_mode == "full":
        sents = [a for r in ins for a in r["answer"]]
        cits = [(a, c) for a in sents for c in a["citations"]]
        exp_by_sentence = {id(a): r["expected"] for r in ins for a in r["answer"]}
        quotes = [q for a in sents for q in a["quotes"]]
        n_valid = sum(len(a["citations"]) for a in sents)
        n_invalid = sum(len(a["invalid_citations"]) for a in sents)
        metrics.update(
            {
                "n_answer_sentences": len(sents),
                "faithfulness_supported": _pct([1.0 if a["faithfulness"] == "Supported" else 0.0 for a in sents]),
                "faithfulness_supported_or_partial": _pct(
                    [1.0 if a["faithfulness"] != "Unsupported" else 0.0 for a in sents]
                ),
                "faithfulness_labels": dict(Counter(a["faithfulness"] for a in sents)),
                "citation_validity": round(n_valid / (n_valid + n_invalid), 3) if n_valid + n_invalid else None,
                "quote_verbatim_rate": _pct([1.0 if q["verbatim"] else 0.0 for q in quotes]),
                "citation_precision_nli": _pct(
                    [1.0 if a["citation_entailment"].get(c, 0.0) >= s.nli_partial else 0.0 for a, c in cits]
                ),
                "citation_precision_expected": _pct(
                    [1.0 if _matches(c, exp_by_sentence[id(a)]) else 0.0 for a, c in cits]
                ),
                "confidence_labels_in_scope": dict(
                    Counter(r["confidence"]["label"] for r in ins if not r["abstained"])
                ),
                "mean_consistency_in_scope": _pct(
                    [r["consistency"]["score"] for r in ins if r["consistency"] and not r["abstained"]]
                ),
            }
        )

    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": run_mode,
        "provider": pipe.provider.name if (pipe.provider and run_mode == "full") else None,
        "model": pipe.provider.model if (pipe.provider and run_mode == "full") else None,
        "consistency_samples": s.consistency_samples if n_samples is None else n_samples,
        "settings": {
            k: getattr(s, k)
            for k in (
                "embedding_model",
                "reranker_model",
                "nli_model",
                "top_k",
                "min_retrieval_strength",
                "nli_supported",
                "nli_partial",
                "w_retrieval",
                "w_faithfulness",
                "w_consistency",
                "conf_high",
                "conf_medium",
                "abstain_below",
            )
        },
        "metric_definitions": __doc__.split("Metric definitions (also written into the results file):")[1].strip(),
        "metrics": metrics,
        "rows": rows,
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--samples", type=int, default=None, help="Override self-consistency samples (full mode).")
    ap.add_argument("--fresh", action="store_true", help="Ignore cached per-question results.")
    ap.add_argument("--replay", action="store_true", help="Re-verify recorded LLM outputs; no API calls.")
    args = ap.parse_args()
    out = evaluate(args.mode, args.samples, fresh=args.fresh or args.replay, replay=args.replay)
    path = ROOT / "eval" / "results" / f"{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["metrics"], indent=2))
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
