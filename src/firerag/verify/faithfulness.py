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


_INLINE_CITATION = re.compile(r"\s*\[(?:v[12]-[^\]]+|chunk_id:[^\]]*)\]")
_TRAILING_QUOTE = re.compile(r"\s*[\"“][^\"”]{3,}[\"”]\.?\s*$")


def clean_hypothesis(sentence: str) -> str:
    """Remove inline chunk-ID brackets and a trailing quoted fragment from a sentence."""
    out = _INLINE_CITATION.sub("", sentence)
    while _TRAILING_QUOTE.search(out):
        out = _TRAILING_QUOTE.sub("", out)
    return out.strip() or sentence


def premise_prefix_for(volume: int, context: str, kind: str, ref: str | None) -> str:
    """Provenance statement prepended to NLI premises, mirroring what the LLM is shown."""
    label = {"table": "Table", "diagram": "Diagram"}.get(kind, "paragraph")
    where = f", {label} {ref}" if ref else ""
    return f"In Approved Document B {context}{where}: "


_LIST_ITEM = re.compile(r"^(?:[a-z]|[ivx]{1,4})\.\s")


def split_windows(text: str, size: int = 3) -> list[str]:
    """Premise windows for NLI.

    1. Overlapping windows of 1 to ``size`` sentences/list items (stride 1).
    2. List-aware windows: Approved Document B states most conditions as a lead-in
       ("... if either of the following applies.") followed by lettered items. NLI
       cannot link item "b." back to its lead-in when they sit apart, so each item
       (with its sub-items) is also paired directly with the sentence introducing it.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+|\n+", text) if p.strip()]
    if not parts:
        return [text]
    # Windows of 1..size sentences. Single sentences matter: the NLI model labels a
    # sentence vs itself as entailment, but adding one unrelated sentence to the
    # premise flips it to "neutral" (found on eval q18).
    windows = [" ".join(parts[i : i + k]) for k in range(1, size + 1) for i in range(max(len(parts) - k + 1, 1))]

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    stem: str | None = None
    for i, line in enumerate(lines):
        if not _LIST_ITEM.match(line):
            stem = line
            continue
        if stem is None or not re.match(r"^[a-z]\.\s", line):
            continue  # roman sub-items are attached to their parent below
        item = [line]
        for nxt in lines[i + 1 :]:
            if re.match(r"^[ivx]{1,4}\.\s", nxt):
                item.append(nxt)
            else:
                break
        windows.append(f"{stem} {' '.join(item)}")
    return list(dict.fromkeys(windows))


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
    premise_prefix: dict[str, str] | None = None,
) -> SentenceFaithfulness:
    """Label one sentence as Supported / Partially supported / Unsupported.

    ``premise_prefix`` maps chunk_id to a provenance statement (volume, section,
    paragraph). Each window is scored both with and without it and the best
    entailment is kept: without the prefix NLI rejects true sentences such as
    "In Volume 1, blocks of flats … should be fitted with sprinklers" (the passage
    never says which volume it is from); with it, some plain restatements score lower.
    """
    sentence = clean_hypothesis(sentence)
    premise_prefix = premise_prefix or {}
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
        prefix = premise_prefix.get(cid, "")
        for w in split_windows(evidence_text[cid]):
            # Score each window with and without the provenance prefix and keep the best:
            # the prefix rescues sentences that name the volume/section, but on eval q01 it
            # cut a verbatim restatement from 0.78 to 0.05. Extra context should only help.
            for premise in dict.fromkeys((w, prefix + w)):
                pairs.append((premise, sentence))
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

    # Deterministic numeric guard. NLI is weak with numbers: in an offline perturbation
    # test it accepted 23% of sentences whose numbers had been doubled ("1100mm" →
    # "2200mm"). In fire-safety guidance the numbers are the guidance, so any number
    # absent from the cited passages (including their provenance) makes the sentence
    # Unsupported, whatever NLI says.
    missing = missing_numbers(sentence, [premise_prefix.get(c, "") + evidence_text[c] for c in cited])
    if missing:
        label = "Unsupported"
        note = f"Number(s) not found in the cited passages: {', '.join(missing)}."
    return SentenceFaithfulness(
        label=label,
        entailment=round(e_best, 4),
        contradiction=round(c_best, 4),
        best_chunk=owners[best],
        per_citation={k: round(v, 4) for k, v in per_cit.items()},
        note=note,
    )


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def missing_numbers(sentence: str, passages: list[str]) -> list[str]:
    """Numbers in ``sentence`` that appear in none of ``passages`` (thousands separators ignored)."""
    norm = [p.replace(",", "") for p in passages]
    found = {n for p in norm for n in _NUMBER.findall(p)}
    return [n for n in dict.fromkeys(_NUMBER.findall(sentence.replace(",", ""))) if n not in found]


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
