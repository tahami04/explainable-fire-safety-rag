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


def save_index(
    index_dir: Path, chunks: list[Chunk], vectors: np.ndarray, provenance: dict[str, dict[str, str]], manifest: dict
) -> None:
    """Persist chunks, FAISS vectors, tag provenance and a manifest."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    fi = faiss.IndexFlatIP(vectors.shape[1])
    fi.add(vectors)
    faiss.write_index(fi, str(index_dir / VECTORS_FILE))
    (index_dir / TAGS_FILE).write_text(json.dumps(provenance, indent=1))
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    """BM25 over structural context + text (so section titles are searchable)."""
    return BM25Okapi([tokenize(c.index_text()) for c in chunks])


def load_index(index_dir: Path) -> Index:
    """Load a saved index; raises FileNotFoundError with a helpful message if missing."""
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {index_dir}. Run: python scripts/download_data.py && python -m firerag.ingest.build_index"
        )
    chunks = [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]
    fi = faiss.read_index(str(index_dir / VECTORS_FILE))
    provenance = json.loads((index_dir / TAGS_FILE).read_text())
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    return Index(chunks, fi, build_bm25(chunks), provenance, manifest)
