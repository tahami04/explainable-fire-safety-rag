"""End-to-end question answering: retrieve → generate → verify → log.

The pipeline always returns the retrieved evidence with its explanations, so the
user can check the sources even when the system abstains or runs without an LLM.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel

from firerag.config import ABSTAIN_MESSAGE, Settings, get_settings
from firerag.generate.generator import generate
from firerag.generate.prompts import PROMPT_VERSION
from firerag.generate.providers import LLMProvider, get_provider
from firerag.generate.schema import ReasoningStep
from firerag.governance.audit import now_iso, write_audit
from firerag.retrieve.hybrid import HybridRetriever, RetrievedChunk, retrieval_strength
from firerag.retrieve.index import load_index
from firerag.verify.caveats import provenance_caveats
from firerag.verify.citations import validate_citations
from firerag.verify.confidence import (
    ConfidenceBreakdown,
    post_generation_abstention,
    retrieval_gate,
    score_confidence,
)
from firerag.verify.consistency import ConsistencyReport, measure_consistency
from firerag.verify.faithfulness import (
    NLIScorer,
    check_sentence,
    default_scorer,
    faithfulness_rate,
    table_fallback,
)

Mode = Literal["full", "retrieval_only"]


class EvidenceItem(BaseModel):
    """A retrieved passage as returned to clients."""

    rank: int
    chunk_id: str
    citation: str
    volume: int
    requirement: str
    section_title: str
    subsection: str | None
    paragraph_ref: str | None
    kind: str
    page: int | None
    pdf_page: int
    effective_note: str | None
    taxonomy: list[str]
    text: str
    explanation: dict


class AnswerSentenceOut(BaseModel):
    """A verified answer sentence."""

    sentence: str
    citations: list[str]
    citation_labels: list[str]
    invalid_citations: list[str]
    quotes: list[dict]
    faithfulness: str
    entailment: float
    faithfulness_note: str
    citation_entailment: dict[str, float] = {}


class AskResult(BaseModel):
    """The full, auditable response to a question."""

    query_id: str
    timestamp: str
    question: str
    mode: Mode
    taxonomy_filter: list[str]
    question_interpretation: str | None
    answer: list[AnswerSentenceOut]
    answer_text: str
    reasoning_pathway: list[ReasoningStep]
    caveats: list[str]
    abstained: bool
    abstention_reason: str | None
    confidence: ConfidenceBreakdown
    consistency: ConsistencyReport | None
    evidence: list[EvidenceItem]
    provider: str | None
    model: str | None
    prompt_version: str
    latency_ms: dict[str, int]


def _evidence_items(results: list[RetrievedChunk]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            rank=r.explanation.final_rank,
            chunk_id=r.chunk.chunk_id,
            citation=r.chunk.citation_label,
            volume=r.chunk.volume,
            requirement=r.chunk.requirement,
            section_title=r.chunk.section_title,
            subsection=r.chunk.subsection,
            paragraph_ref=r.chunk.paragraph_ref,
            kind=r.chunk.kind,
            page=r.chunk.page,
            pdf_page=r.chunk.pdf_page,
            effective_note=r.chunk.effective_note,
            taxonomy=r.chunk.taxonomy,
            text=r.chunk.text,
            explanation=r.explanation.model_dump(),
        )
        for r in results
    ]


class Pipeline:
    """Holds the loaded index and models; call :meth:`ask`."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        provider: LLMProvider | None | Literal["auto"] = "auto",
        nli_scorer: NLIScorer | None = None,
        embed: Callable[[list[str]], np.ndarray] | None = None,
        audit: bool = True,
    ) -> None:
        """Create a pipeline; dependencies are injectable for tests and evaluation."""
        self.s = settings or get_settings()
        self.retriever = retriever or HybridRetriever(load_index(self.s.index_dir), self.s)
        self.provider = get_provider(self.s) if provider == "auto" else provider
        self._nli = nli_scorer
        self._embed = embed
        self.audit = audit

    @property
    def nli(self) -> NLIScorer:
        """NLI scorer, loaded on first use."""
        if self._nli is None:
            self._nli = default_scorer()
        return self._nli

    @property
    def embed(self) -> Callable[[list[str]], np.ndarray]:
        """Embedding function for consistency checks, loaded on first use."""
        if self._embed is None:
            from firerag.retrieve.models import embed_passages

            self._embed = embed_passages
        return self._embed

    def ask(
        self,
        question: str,
        taxonomy_filter: list[str] | None = None,
        mode: Mode | None = None,
        n_samples: int | None = None,
    ) -> AskResult:
        """Answer ``question`` with full provenance, verification and audit logging."""
        s = self.s
        t0 = time.perf_counter()
        timings: dict[str, int] = {}
        taxonomy_filter = taxonomy_filter or []
        mode = mode or ("full" if self.provider is not None else "retrieval_only")
        if mode == "full" and self.provider is None:
            mode = "retrieval_only"
        n_samples = s.consistency_samples if n_samples is None else n_samples

        results = self.retriever.search(question, top_k=s.top_k, taxonomy_filter=taxonomy_filter)
        timings["retrieval"] = int((time.perf_counter() - t0) * 1000)
        strength = retrieval_strength(results)
        evidence = _evidence_items(results)
        gate = retrieval_gate(strength, s)

        base = dict(
            query_id=str(uuid.uuid4()),
            timestamp=now_iso(),
            question=question,
            mode=mode,
            taxonomy_filter=taxonomy_filter,
            evidence=evidence,
            provider=self.provider.name if (self.provider and mode == "full") else None,
            model=self.provider.model if (self.provider and mode == "full") else None,
            prompt_version=PROMPT_VERSION,
        )

        if mode == "retrieval_only" or gate.abstain:
            conf = score_confidence(strength, None, None, 0, s)
            if mode == "retrieval_only":
                conf = conf.model_copy(
                    update={
                        "overall": None,
                        "label": "n/a",
                        "reasons": [
                            conf.reasons[0],
                            "Retrieval-only mode: no answer was generated, so only evidence relevance is scored.",
                        ],
                    }
                )
            result = AskResult(
                **base,
                question_interpretation=None,
                answer=[],
                answer_text=ABSTAIN_MESSAGE if gate.abstain else "",
                reasoning_pathway=[],
                caveats=[],
                abstained=gate.abstain,
                abstention_reason=gate.reason,
                confidence=conf,
                consistency=None,
                latency_ms={**timings, "total": int((time.perf_counter() - t0) * 1000)},
            )
            self._log(result)
            return result

        t1 = time.perf_counter()
        gen = generate(self.provider, question, results, n_samples, s.sample_temperature)
        timings["generation"] = int((time.perf_counter() - t1) * 1000)
        if gen.primary is None:
            conf = score_confidence(strength, None, None, 0, s)
            result = AskResult(
                **base,
                question_interpretation=None,
                answer=[],
                answer_text=ABSTAIN_MESSAGE,
                reasoning_pathway=[],
                caveats=["The model returned malformed output."],
                abstained=True,
                abstention_reason="Generation failed validation.",
                confidence=conf,
                consistency=None,
                latency_ms={**timings, "total": int((time.perf_counter() - t0) * 1000)},
            )
            self._log(result)
            return result

        t2 = time.perf_counter()
        primary = gen.primary
        ev_text = {r.chunk.chunk_id: r.chunk.text for r in results}
        by_id = {r.chunk.chunk_id: r.chunk for r in results}
        cit_reports = validate_citations(primary, ev_text)

        sentences: list[AnswerSentenceOut] = []
        labels = []
        for sent, rep in zip(primary.answer, cit_reports, strict=True):
            f = check_sentence(sent.sentence, rep.valid, ev_text, self.nli, s.nli_supported, s.nli_partial)
            tables = {c: ev_text[c] for c in rep.valid if by_id[c].kind == "table"}
            f = table_fallback(f, sent.sentence, tables, rep.all_quotes_verbatim, bool(rep.quotes))
            note = f.note
            if not rep.all_quotes_verbatim:
                note = (note + " " if note else "") + "A quote was not found verbatim in its cited passage."
            labels.append(f.label)
            sentences.append(
                AnswerSentenceOut(
                    sentence=sent.sentence,
                    citations=rep.valid,
                    citation_labels=[by_id[c].citation_label for c in rep.valid],
                    invalid_citations=rep.invalid,
                    quotes=[q.model_dump() for q in rep.quotes],
                    faithfulness=f.label,
                    entailment=f.entailment,
                    faithfulness_note=note.strip(),
                    citation_entailment=f.per_citation,
                )
            )
        timings["verification"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        cons = measure_consistency(primary, gen.samples, self.embed) if gen.samples else None
        timings["consistency"] = int((time.perf_counter() - t3) * 1000)

        faith = faithfulness_rate(labels) if labels else None
        n_unsupported = sum(lab == "Unsupported" for lab in labels)
        conf = score_confidence(strength, faith, cons.score if cons else None, n_unsupported, s)
        n_valid = sum(1 for r in cit_reports if r.has_valid_citation)
        decision = post_generation_abstention(primary.abstained, n_valid, conf.overall or 0.0, s, faithfulness=faith)

        cited_ids = list(dict.fromkeys(c for so in sentences for c in so.citations))
        caveats = list(dict.fromkeys(primary.caveats + provenance_caveats([by_id[c] for c in cited_ids])))
        valid_ids = set(ev_text)
        pathway = [
            ReasoningStep(step=st.step, evidence=[e for e in st.evidence if e in valid_ids])
            for st in primary.reasoning_pathway
        ]

        result = AskResult(
            **base,
            question_interpretation=primary.question_interpretation,
            answer=[] if decision.abstain else sentences,
            answer_text=ABSTAIN_MESSAGE if decision.abstain else " ".join(x.sentence for x in sentences),
            reasoning_pathway=[] if decision.abstain else pathway,
            caveats=caveats,
            abstained=decision.abstain,
            abstention_reason=decision.reason,
            confidence=conf,
            consistency=cons,
            latency_ms={**timings, "total": int((time.perf_counter() - t0) * 1000)},
        )
        # Keep the withheld draft in the audit trail so abstentions can be reviewed.
        self._log(result, withheld=sentences if decision.abstain and sentences else None)
        return result

    def _log(self, r: AskResult, withheld: list[AnswerSentenceOut] | None = None) -> None:
        if not self.audit:
            return
        write_audit(
            self.s.log_dir,
            {
                "query_id": r.query_id,
                "timestamp": r.timestamp,
                "question": r.question,
                "mode": r.mode,
                "taxonomy_filter": r.taxonomy_filter,
                "retrieved": [
                    {
                        "chunk_id": e.chunk_id,
                        "rank": e.rank,
                        "rerank_relevance": e.explanation.get("rerank_relevance"),
                        "bm25_score": e.explanation.get("bm25_score"),
                        "dense_similarity": e.explanation.get("dense_similarity"),
                    }
                    for e in r.evidence
                ],
                "provider": r.provider,
                "model": r.model,
                "prompt_version": r.prompt_version,
                "answer": [
                    {"sentence": a.sentence, "citations": a.citations, "faithfulness": a.faithfulness} for a in r.answer
                ],
                "withheld_draft": [
                    {"sentence": a.sentence, "citations": a.citations, "faithfulness": a.faithfulness} for a in withheld
                ]
                if withheld
                else None,
                "confidence": r.confidence.model_dump(),
                "consistency": r.consistency.model_dump() if r.consistency else None,
                "abstained": r.abstained,
                "abstention_reason": r.abstention_reason,
                "latency_ms": r.latency_ms,
            },
        )
