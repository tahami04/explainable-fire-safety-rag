"""On-disk index: chunks, dense vectors (FAISS) and tagging provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from firerag.models import Chunk
from firerag.retrieve.text import tokenize

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.faiss"
TAGS_FILE = "taxonomy_provenance.json"
MANIFEST_FILE = "manifest.json"


@dataclass
class Index:
    """Everything retrieval needs, loaded into memory."""

    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    tag_provenance: dict[str, dict[str, str]]
    manifest: dict

    @property
    def by_id(self) -> dict[str, Chunk]:
        """Chunks keyed by ``chunk_id``."""
        return {c.chunk_id: c for c in self.chunks}


def make_faiss(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index over L2-normalised vectors (inner product = cosine).

    Always import faiss through this package: ``firerag/__init__`` loads torch first,
    which avoids a macOS OpenMP crash when faiss is loaded before torch.
    """
    fi = faiss.IndexFlatIP(vectors.shape[1])
    fi.add(vectors)
    return fi


def save_index(
    index_dir: Path, chunks: list[Chunk], vectors: np.ndarray, provenance: dict[str, dict[str, str]], manifest: dict
) -> None:
    """Persist chunks, FAISS vectors, tag provenance and a manifest."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    faiss.write_index(make_faiss(vectors), str(index_dir / VECTORS_FILE))
    (index_dir / TAGS_FILE).write_text(json.dumps(provenance, indent=1))
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    """BM25 over structural context + text (so section titles are searchable)."""
    return BM25Okapi([tokenize(c.index_text()) for c in chunks])


def verify_faiss(fi: faiss.Index, n_checks: int = 5) -> None:
    """Check faiss search results against exact NumPy inner products.

    Guards the macOS OpenMP workaround in ``firerag/__init__``: if faiss ever returned
    wrong neighbours we would rather fail loudly than show the wrong evidence.
    """
    vecs = fi.reconstruct_n(0, fi.ntotal)
    rng = np.random.default_rng(0)
    for i in rng.choice(fi.ntotal, size=min(n_checks, fi.ntotal), replace=False):
        k = min(10, fi.ntotal)
        _, ids = fi.search(vecs[i : i + 1], k)
        exact = np.argsort(-(vecs @ vecs[i]), kind="stable")[:k]
        if set(ids[0].tolist()) != set(exact.tolist()):
            raise RuntimeError("FAISS self-check failed: results differ from exact NumPy search.")


def load_index(index_dir: Path) -> Index:
    """Load a saved index; raises FileNotFoundError with a helpful message if missing."""
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {index_dir}. Run: python scripts/download_data.py && python -m firerag.ingest.build_index"
        )
    chunks = [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]
    fi = faiss.read_index(str(index_dir / VECTORS_FILE))
    verify_faiss(fi)
    provenance = json.loads((index_dir / TAGS_FILE).read_text())
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    return Index(chunks, fi, build_bm25(chunks), provenance, manifest)
