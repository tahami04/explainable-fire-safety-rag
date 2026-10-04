"""Overall confidence and abstention.

The overall score is a transparent weighted sum of three signals, each shown to
the user, plus hard rules that can only *lower* confidence:

    overall = w_r · retrieval_strength + w_f · faithfulness_rate + w_c · consistency

* High   : overall ≥ conf_high, faithfulness ≥ 0.75 and no Unsupported sentence
* Medium : overall ≥ conf_medium and faithfulness ≥ 0.5
* Low    : otherwise
* Abstain: overall < abstain_below, or faithfulness is 0 (no sentence even partly
           supported), or the model abstained, or nothing survived citation
           validation, or retrieval was too weak to attempt an answer.

Strong retrieval and high self-consistency can never compensate for an answer
that its own citations do not support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from firerag.config import Settings

ConfidenceLabel = Literal["High", "Medium", "Low", "n/a"]
MIN_FAITHFULNESS_FOR_MEDIUM = 0.5
MIN_FAITHFULNESS_FOR_HIGH = 0.75


class ConfidenceBreakdown(BaseModel):
    """All inputs and outputs of the confidence decision, for display and audit."""

    retrieval_strength: float
    faithfulness_rate: float | None
    consistency: float | None
    weights: dict[str, float]
    overall: float | None
    label: ConfidenceLabel
    reasons: list[str]


class AbstentionDecision(BaseModel):
    """Whether to withhold the answer, and why."""

    abstain: bool
    reason: str | None = None


def retrieval_gate(strength: float, s: Settings) -> AbstentionDecision:
    """Pre-generation gate: is any passage relevant enough to attempt an answer?"""
    if strength < s.min_retrieval_strength:
        return AbstentionDecision(
            abstain=True,
            reason=(
                f"Best passage relevance {strength:.2f} is below the threshold "
                f"{s.min_retrieval_strength:.2f}; the question looks out of scope."
            ),
        )
    return AbstentionDecision(abstain=False)


def score_confidence(
    retrieval_strength: float, faithfulness: float | None, consistency: float | None, n_unsupported: int, s: Settings
) -> ConfidenceBreakdown:
    """Combine the signals into an overall score and a High/Medium/Low label."""
    weights = {"retrieval": s.w_retrieval, "faithfulness": s.w_faithfulness, "consistency": s.w_consistency}
    parts = {"retrieval": retrieval_strength, "faithfulness": faithfulness, "consistency": consistency}
    used = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(weights[k] for k in used)
    overall = sum(weights[k] * v for k, v in used.items()) / total_w if total_w else 0.0

    reasons = [f"Retrieval strength {retrieval_strength:.2f} (best reranker relevance of the evidence)."]
    if faithfulness is not None:
        reasons.append(f"Faithfulness {faithfulness:.2f} (share of sentences entailed by their citations).")
    if consistency is not None:
        reasons.append(f"Self-consistency {consistency:.2f} (agreement across sampled answers).")
    else:
        reasons.append("Self-consistency not measured; weights re-normalised over the other signals.")

    if faithfulness is not None and faithfulness < MIN_FAITHFULNESS_FOR_MEDIUM:
        label: ConfidenceLabel = "Low"
        if overall >= s.conf_medium:
            reasons.append(
                f"Capped at Low: faithfulness {faithfulness:.2f} is below {MIN_FAITHFULNESS_FOR_MEDIUM:.2f}."
            )
    elif overall >= s.conf_high and n_unsupported == 0 and (faithfulness or 0.0) >= MIN_FAITHFULNESS_FOR_HIGH:
        label = "High"
    elif overall >= s.conf_medium:
        label = "Medium"
        if overall >= s.conf_high and n_unsupported:
            reasons.append(f"Capped at Medium: {n_unsupported} sentence(s) unsupported by their citations.")
        elif overall >= s.conf_high:
            reasons.append(
                f"Capped at Medium: faithfulness below {MIN_FAITHFULNESS_FOR_HIGH:.2f} "
                "(e.g. table look-ups that NLI cannot fully verify)."
            )
    else:
        label = "Low"
    return ConfidenceBreakdown(
        retrieval_strength=round(retrieval_strength, 4),
        faithfulness_rate=None if faithfulness is None else round(faithfulness, 4),
        consistency=None if consistency is None else round(consistency, 4),
        weights=weights,
        overall=round(overall, 4),
        label=label,
        reasons=reasons,
    )


def post_generation_abstention(
    model_abstained: bool, n_valid_sentences: int, overall: float, s: Settings, faithfulness: float | None = None
) -> AbstentionDecision:
    """Decide whether to withhold a generated answer."""
    if model_abstained:
        return AbstentionDecision(abstain=True, reason="The model judged the evidence insufficient.")
    if n_valid_sentences == 0:
        return AbstentionDecision(abstain=True, reason="No answer sentence had a valid citation.")
    if faithfulness is not None and faithfulness == 0.0:
        return AbstentionDecision(
            abstain=True, reason="No answer sentence is supported, even partly, by its citations."
        )
    if overall < s.abstain_below:
        return AbstentionDecision(
            abstain=True,
            reason=f"Overall confidence {overall:.2f} is below the abstention threshold {s.abstain_below:.2f}.",
        )
    return AbstentionDecision(abstain=False)
