"""Lazy loaders for local transformer models (embeddings, reranker, NLI).

Loading is deferred so that tests, the API schema and keyless mode never pay for
model downloads they do not need.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from firerag.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    """The sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model, device="cpu")


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    """The cross-encoder relevance reranker."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model, device="cpu")


@lru_cache(maxsize=1)
def nli_model() -> CrossEncoder:
    """The NLI cross-encoder used for faithfulness checks."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().nli_model, device="cpu")


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalised passage embeddings."""
    vecs = embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_queries(texts: list[str]) -> np.ndarray:
    """L2-normalised query embeddings (bge models expect an instruction prefix)."""
    prefix = BGE_QUERY_PREFIX if "bge" in get_settings().embedding_model.lower() else ""
    return embed_passages([prefix + t for t in texts])
