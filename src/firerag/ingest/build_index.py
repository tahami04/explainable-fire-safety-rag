"""Build the retrieval index: parse → chunk → repair → tag → embed → save.

Usage:
    python -m firerag.ingest.build_index
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from firerag.config import get_settings
from firerag.ingest.chunker import chunk_volume, repair_ligatures
from firerag.retrieve.index import save_index
from firerag.retrieve.models import embed_passages
from firerag.taxonomy.tagger import load_taxonomy, tag_chunks

VOLUMES = {1: "AD_B_vol1_dwellings.pdf", 2: "AD_B_vol2_non_dwellings.pdf"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Build and save the index under ``settings.index_dir``."""
    s = get_settings()
    raw = s.data_dir / "raw"
    chunks = []
    for vol, name in VOLUMES.items():
        pdf = raw / name
        if not pdf.exists():
            raise SystemExit(f"Missing {pdf}. Run: python scripts/download_data.py")
        vol_chunks = chunk_volume(pdf, vol)
        print(f"Volume {vol}: {len(vol_chunks)} chunks")
        chunks.extend(vol_chunks)

    chunks, repairs = repair_ligatures(chunks)
    print(f"Ligature repairs applied: {len(repairs)} word forms")

    vectors = embed_passages([c.index_text() for c in chunks])
    tax = load_taxonomy(s.taxonomy_path)
    chunks, provenance = tag_chunks(chunks, tax, embed=embed_passages, chunk_vectors=vectors)

    rule_counts = Counter(r.split(" ")[0] for tags in provenance.values() for r in tags.values())
    untagged = sum(1 for c in chunks if not any("." in t for t in c.taxonomy))
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {name: _sha256(raw / name) for name in VOLUMES.values()},
        "embedding_model": s.embedding_model,
        "n_chunks": len(chunks),
        "kinds": Counter(c.kind for c in chunks),
        "ligature_repairs": repairs,
        "taxonomy_version": tax.version,
        "tag_rule_counts": rule_counts,
        "chunks_without_subtopic": untagged,
    }
    save_index(s.index_dir, chunks, vectors, provenance, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "ligature_repairs"}, indent=2))


if __name__ == "__main__":
    main()
