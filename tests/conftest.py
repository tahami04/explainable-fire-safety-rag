"""Shared offline fixtures: a tiny in-memory index, fake embedder and fake LLM.

Nothing here downloads a model or calls a network API.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from firerag.config import Settings
from firerag.models import Chunk
from firerag.retrieve.hybrid import HybridRetriever
from firerag.retrieve.index import Index, build_bm25, make_faiss
from firerag.retrieve.text import tokenize

DIM = 64


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words vectors, L2-normalised."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


CHUNKS = [
    Chunk(
        chunk_id="v2-2.3",
        volume=2,
        requirement="B1",
        section_title="Section 2: Design for horizontal escape",
        subsection="Escape route design",
        paragraph_ref="2.3",
        page=12,
        pdf_page=24,
        taxonomy=["B1", "B1.escape_routes"],
        text="The number of escape routes and exits that should be provided depends on the number of occupants "
        "and the limits on travel distance to the nearest exit given in Table 2.1.",
    ),
    Chunk(
        chunk_id="v2-8.6",
        volume=2,
        requirement="B3",
        section_title="Section 8: Compartmentation/sprinklers",
        subsection="Protected shafts",
        paragraph_ref="8.6",
        page=65,
        pdf_page=77,
        taxonomy=["B3", "B3.compartmentation"],
        text="Stairs and service shafts connecting compartments should be protected to restrict the spread of fire "
        "between the compartments. These are called protected shafts.",
    ),
    Chunk(
        chunk_id="v1-B10~2",
        volume=1,
        requirement="Appendix",
        section_title="Appendix B: Performance of materials, products and structures",
        paragraph_ref="B10",
        page=138,
        pdf_page=150,
        effective_note="2029 amendment – effective from 2 September 2029",
        taxonomy=["B2", "B2.thermoplastics"],
        text="Thermoplastic material is any synthetic polymeric material that has a softening point below 200°C.",
    ),
]


def make_index(chunks: list[Chunk] = CHUNKS) -> Index:
    vecs = fake_embed([c.index_text() for c in chunks])
    return Index(chunks, make_faiss(vecs), build_bm25(chunks), {}, {})


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        log_dir=tmp_path / "logs",
        use_reranker=False,
        llm_provider="none",
        openai_api_key=None,
        consistency_samples=2,
    )


@pytest.fixture
def retriever(settings) -> HybridRetriever:
    return HybridRetriever(make_index(), settings, embed_query=lambda q: fake_embed([q])[0], rerank=None)


class FakeLLM:
    """Returns a scripted JSON answer for every call."""

    name = "fake"
    model = "scripted"

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[float, int]] = []

    def complete(self, messages, temperature, n=1):
        self.calls.append((temperature, n))
        return [json.dumps(self.payload)] * n


def entail_all(pairs):
    """Fake NLI scorer: every pair is strongly entailed (logits in contradiction/entailment/neutral order)."""
    return np.tile([-5.0, 5.0, -5.0], (len(pairs), 1))


def entail_none(pairs):
    """Fake NLI scorer: every pair is neutral."""
    return np.tile([-5.0, -5.0, 5.0], (len(pairs), 1))
