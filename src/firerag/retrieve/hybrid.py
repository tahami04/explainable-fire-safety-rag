"""Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion, reranking and
per-result explanations.

Every result carries the evidence for *why* it was retrieved: its BM25 score and
rank, dense cosine similarity and rank, RRF score, reranker relevance, and the
query terms it matched.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from firerag.config import Settings, get_settings
from firerag.models import Chunk
from firerag.retrieve.index import Index
from firerag.retrieve.text import highlight_spans, matched_terms, tokenize


class RetrievalExplanation(BaseModel):
    """Why a chunk was retrieved."""

    bm25_score: float
    bm25_rank: int | None  # None = not in BM25 top candidates
    dense_similarity: float
    dense_rank: int | None
    rrf_score: float
    fused_rank: int
    rerank_relevance: float | None  # sigmoid of cross-encoder logit, 0–1
    final_rank: int
    matched_terms: list[str]
    highlight_spans: list[tuple[int, int]]


class RetrievedChunk(BaseModel):
    """A retrieved chunk plus its retrieval explanation."""

    chunk: Chunk
    explanation: RetrievalExplanation


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_r(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return scores


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    """BM25 + dense retrieval over an :class:`Index`."""

    def __init__(self, index: Index, settings: Settings | None = None, embed_query=None, rerank=None) -> None:
        """Create a retriever.

        ``embed_query`` and ``rerank`` are injectable for testing; by default the
        local sentence-transformers models are used.
        """
        self.index = index
        self.s = settings or get_settings()
        if embed_query is None:
            from firerag.retrieve.models import embed_queries

            embed_query = lambda q: embed_queries([q])[0]  # noqa: E731
        if rerank is None and self.s.use_reranker:
            from firerag.retrieve.models import reranker

            rerank = lambda pairs: reranker().predict(pairs, show_progress_bar=False)  # noqa: E731
        self._embed_query = embed_query
        self._rerank = rerank
        self._index_texts = [c.index_text() for c in index.chunks]

    def _allowed(self, taxonomy_filter: list[str] | None) -> np.ndarray:
        n = len(self.index.chunks)
        if not taxonomy_filter:
            return np.ones(n, dtype=bool)
        wanted = set(taxonomy_filter)
        return np.array([bool(wanted & set(c.taxonomy)) for c in self.index.chunks])

    def search(
        self, query: str, top_k: int | None = None, taxonomy_filter: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks for ``query`` with explanations."""
        s = self.s
        top_k = top_k or s.top_k
        allowed = self._allowed(taxonomy_filter)
        n_cand = s.candidates_per_retriever

        bm25_scores = np.asarray(self.index.bm25.get_scores(tokenize(query)), dtype=float)
        bm25_scores = np.where(allowed, bm25_scores, -np.inf)
        bm25_order = [
            int(i) for i in np.argsort(-bm25_scores)[:n_cand] if np.isfinite(bm25_scores[i]) and bm25_scores[i] > 0
        ]

        qv = np.asarray(self._embed_query(query), dtype=np.float32).reshape(1, -1)
        sims, ids = self.index.faiss_index.search(qv, len(self.index.chunks))
        dense_sim = np.empty(len(self.index.chunks), dtype=float)
        dense_sim[ids[0]] = sims[0]
        dense_order = [int(i) for i in ids[0] if allowed[i]][:n_cand]

        fused = rrf([bm25_order, dense_order], k=s.rrf_k)
        fused_order = sorted(fused, key=lambda d: -fused[d])
        bm25_rank = {d: r for r, d in enumerate(bm25_order, start=1)}
        dense_rank = {d: r for r, d in enumerate(dense_order, start=1)}

        pool = fused_order[: s.rerank_pool]
        rel: dict[int, float] = {}
        if self._rerank is not None and pool:
            logits = self._rerank([(query, self._index_texts[d]) for d in pool])
            rel = {d: _sigmoid(float(lg)) for d, lg in zip(pool, logits, strict=True)}
            final = sorted(pool, key=lambda d: -rel[d])
        else:
            final = pool

        out: list[RetrievedChunk] = []
        for final_rank, d in enumerate(final[:top_k], start=1):
            chunk = self.index.chunks[d]
            terms = matched_terms(query, chunk.text)
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    explanation=RetrievalExplanation(
                        bm25_score=round(float(max(bm25_scores[d], 0.0)), 3),
                        bm25_rank=bm25_rank.get(d),
                        dense_similarity=round(float(dense_sim[d]), 4),
                        dense_rank=dense_rank.get(d),
                        rrf_score=round(fused[d], 5),
                        fused_rank=fused_order.index(d) + 1,
                        rerank_relevance=round(rel[d], 4) if d in rel else None,
                        final_rank=final_rank,
                        matched_terms=terms,
                        highlight_spans=highlight_spans(chunk.text, terms),
                    ),
                )
            )
        return out


def retrieval_strength(results: list[RetrievedChunk]) -> float:
    """Best relevance among results (reranker probability, else dense similarity)."""
    if not results:
        return 0.0
    e = results[0].explanation
    if e.rerank_relevance is not None:
        return max(r.explanation.rerank_relevance or 0.0 for r in results)
    return max(0.0, max(r.explanation.dense_similarity for r in results))
