"""Structured output schema the LLM must return, validated with Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Quote(BaseModel):
    """A short verbatim extract from a cited passage."""

    chunk_id: str
    text: str


class AnswerSentence(BaseModel):
    """One answer sentence with the evidence it relies on."""

    sentence: str
    citations: list[str] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    """One step of the evidence-linked reasoning summary."""

    step: str
    evidence: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """The model's structured answer.

    ``reasoning_pathway`` is an *evidence-linked reasoning summary* written for the
    reader. It is not, and is not presented as, the model's hidden internal reasoning.
    """

    question_interpretation: str
    answer: list[AnswerSentence] = Field(default_factory=list)
    reasoning_pathway: list[ReasoningStep] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    abstained: bool = False

    def text(self) -> str:
        """The answer as plain prose."""
        return " ".join(s.sentence for s in self.answer)

    def cited_ids(self) -> set[str]:
        """All chunk IDs cited anywhere in the answer."""
        return {cid for s in self.answer for cid in s.citations}
