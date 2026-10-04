"""Self-consistency: if the model is asked the same question several times at
temperature > 0, do the answers agree in meaning and in the evidence they cite?

Consistency is not correctness (a model can be consistently wrong), but
disagreement is a cheap, useful warning sign of an under-determined answer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

from firerag.generate.schema import GroundedAnswer

# bge-small cosine between two *unrelated* answers about fire safety is typically
# ~0.6, so raw cosine is rescaled from [SEM_FLOOR, 1] to [0, 1].
SEM_FLOOR = 0.6

Embed = Callable[[list[str]], np.ndarray]


class ConsistencyReport(BaseModel):
    """Agreement between the primary answer and N sampled answers."""

    n_samples: int
    semantic_similarity: float  # mean raw cosine, primary vs each sample
    citation_overlap: float  # mean Jaccard of cited chunk sets
    abstention_agreement: float  # share of samples agreeing with the primary's abstain decision
    score: float  # combined 0–1


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity; two empty sets agree perfectly."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def measure_consistency(primary: GroundedAnswer, samples: list[GroundedAnswer], embed: Embed) -> ConsistencyReport:
    """Compare ``primary`` with each sample."""
    if not samples:
        return ConsistencyReport(
            n_samples=0, semantic_similarity=0.0, citation_overlap=0.0, abstention_agreement=0.0, score=0.0
        )
    abst = float(np.mean([s.abstained == primary.abstained for s in samples]))
    cit = float(np.mean([jaccard(primary.cited_ids(), s.cited_ids()) for s in samples]))
    texts = [primary.text() or "(abstained)"] + [s.text() or "(abstained)" for s in samples]
    vecs = embed(texts)
    sem = float(np.mean(vecs[1:] @ vecs[0]))
    sem_scaled = float(np.clip((sem - SEM_FLOOR) / (1 - SEM_FLOOR), 0.0, 1.0))
    score = abst * (0.5 * sem_scaled + 0.5 * cit)
    return ConsistencyReport(
        n_samples=len(samples),
        semantic_similarity=round(sem, 4),
        citation_overlap=round(cit, 4),
        abstention_agreement=round(abst, 4),
        score=round(score, 4),
    )
