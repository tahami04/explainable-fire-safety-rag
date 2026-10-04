"""Citation validation: cited IDs must be among the retrieved evidence, and quotes
must appear verbatim in the passage they claim to come from.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from firerag.generate.schema import GroundedAnswer

_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-"})


def normalise_for_match(text: str) -> str:
    """Normalise whitespace and typographic quotes/dashes only.

    This is deliberately narrow: wording, numbers and word order must match
    exactly. Case is ignored because models often capitalise a quote that starts
    mid-sentence.
    """
    return re.sub(r"\s+", " ", text.translate(_TYPOGRAPHY)).strip().lower()


class QuoteCheck(BaseModel):
    """Result of checking one quote."""

    chunk_id: str
    text: str
    verbatim: bool


class SentenceCitations(BaseModel):
    """Citation validation result for one answer sentence."""

    valid: list[str]
    invalid: list[str]  # IDs not among the retrieved evidence (removed)
    quotes: list[QuoteCheck]

    @property
    def has_valid_citation(self) -> bool:
        """At least one cited ID exists in the evidence."""
        return bool(self.valid)

    @property
    def all_quotes_verbatim(self) -> bool:
        """Every quote was found verbatim (vacuously true with no quotes)."""
        return all(q.verbatim for q in self.quotes)


def validate_citations(answer: GroundedAnswer, evidence_text: dict[str, str]) -> list[SentenceCitations]:
    """Validate every sentence's citations and quotes against ``evidence_text``.

    ``evidence_text`` maps each retrieved chunk_id to its text.
    """
    norm = {cid: normalise_for_match(t) for cid, t in evidence_text.items()}
    out = []
    for sent in answer.answer:
        cited = list(dict.fromkeys(sent.citations))
        valid = [c for c in cited if c in norm]
        invalid = [c for c in cited if c not in norm]
        checks = []
        for q in sent.quotes:
            ok = q.chunk_id in norm and normalise_for_match(q.text) in norm[q.chunk_id]
            checks.append(QuoteCheck(chunk_id=q.chunk_id, text=q.text, verbatim=ok))
        out.append(SentenceCitations(valid=valid, invalid=invalid, quotes=checks))
    return out
