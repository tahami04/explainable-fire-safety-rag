"""Faithfulness check with an NLI cross-encoder.

For each answer sentence (the hypothesis) we ask: is it *entailed* by the passages
it cites (the premise)? Passages are split into overlapping sentence windows
because NLI models are trained on short premises and truncate long ones; the
sentence's score is the best entailment probability over all windows of all
cited passages.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

FaithLabel = Literal["Supported", "Partially supported", "Unsupported"]

# Label order of cross-encoder/nli-deberta-v3-* models.
NLI_LABELS = ("contradiction", "entailment", "neutral")

NLIScorer = Callable[[list[tuple[str, str]]], np.ndarray]
"""Maps (premise, hypothesis) pairs to an (n, 3) array of logits in NLI_LABELS order."""


class SentenceFaithfulness(BaseModel):
    """Faithfulness verdict for one sentence."""

    label: FaithLabel
    entailment: float  # best entailment probability over cited passages
    contradiction: float  # contradiction probability for that same window
    best_chunk: str | None
    per_citation: dict[str, float]  # chunk_id -> best entailment probability
    note: str = ""


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_windows(text: str, size: int = 3) -> list[str]:
    """Overlapping windows of ``size`` sentences/list items (stride 1)."""
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+|\n+", text) if p.strip()]
    if len(parts) <= size:
        return [" ".join(parts)] if parts else [text]
    return [" ".join(parts[i : i + size]) for i in range(len(parts) - size + 1)]


def default_scorer() -> NLIScorer:
    """The local NLI cross-encoder (loaded lazily)."""
    from firerag.retrieve.models import nli_model

    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(nli_model().predict(pairs, batch_size=32, show_progress_bar=False))

    return score


def check_sentence(
    sentence: str,
    cited: Sequence[str],
    evidence_text: dict[str, str],
    scorer: NLIScorer,
    supported: float,
    partial: float,
) -> SentenceFaithfulness:
    """Label one sentence as Supported / Partially supported / Unsupported."""
    if not cited:
        return SentenceFaithfulness(
            label="Unsupported",
            entailment=0.0,
            contradiction=0.0,
            best_chunk=None,
            per_citation={},
            note="No valid citation.",
        )
    pairs, owners = [], []
    for cid in cited:
        for w in split_windows(evidence_text[cid]):
            pairs.append((w, sentence))
            owners.append(cid)
    probs = _softmax(scorer(pairs))
    ent, con = probs[:, NLI_LABELS.index("entailment")], probs[:, NLI_LABELS.index("contradiction")]
    per_cit: dict[str, float] = {}
    for cid, e in zip(owners, ent, strict=True):
        per_cit[cid] = max(per_cit.get(cid, 0.0), float(e))
    best = int(np.argmax(ent))
    e_best, c_best = float(ent[best]), float(con[best])

    if e_best >= supported:
        label: FaithLabel = "Supported"
        note = ""
    elif e_best >= partial:
        label, note = "Partially supported", "Cited text only partly entails this sentence."
    else:
        label, note = "Unsupported", "Cited text does not entail this sentence."
    if c_best > 0.5 and label != "Supported":
        note = "Cited text appears to contradict this sentence."
    return SentenceFaithfulness(
        label=label,
        entailment=round(e_best, 4),
        contradiction=round(c_best, 4),
        best_chunk=owners[best],
        per_citation={k: round(v, 4) for k, v in per_cit.items()},
        note=note,
    )


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_grounded(sentence: str, passage: str) -> bool:
    """True if the sentence contains numbers and every one appears in the passage."""
    nums = set(_NUMBER.findall(sentence))
    return bool(nums) and nums <= set(_NUMBER.findall(passage))


def table_fallback(
    result: SentenceFaithfulness, sentence: str, cited_tables: dict[str, str], quotes_verbatim: bool, has_quote: bool
) -> SentenceFaithfulness:
    """Upgrade an unsupported sentence that relies on a *table* to Partially supported
    when its quote is verbatim and all its numbers appear in the cited table.

    NLI models are trained on prose and cannot read flattened table rows such as
    "3 | Office | 18 | 45", so they under-score correct table look-ups. This rule
    never upgrades to Supported and always says why.
    """
    if result.label != "Unsupported" or not cited_tables or not (has_quote and quotes_verbatim):
        return result
    for cid, text in cited_tables.items():
        if numbers_grounded(sentence, text):
            return result.model_copy(
                update={
                    "label": "Partially supported",
                    "best_chunk": cid,
                    "note": "Table evidence: NLI cannot read table rows; quote verified verbatim and every "
                    "number in the sentence appears in the cited table.",
                }
            )
    return result


def faithfulness_rate(labels: Sequence[FaithLabel]) -> float:
    """Share of supported sentences, counting partial support as half."""
    if not labels:
        return 0.0
    score = sum(1.0 if lab == "Supported" else 0.5 if lab == "Partially supported" else 0.0 for lab in labels)
    return score / len(labels)
