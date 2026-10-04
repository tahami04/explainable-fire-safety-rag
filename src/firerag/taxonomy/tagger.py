"""Tag chunks with nodes from the draft fire-safety taxonomy.

Rules run in a fixed order (structure → keyword → embedding) and each tag records
which rule produced it, so every tag shown in the UI can be explained.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from firerag.models import Chunk

# Embedding fallback is deliberately conservative: a spot-check at 0.62 with no margin
# rule mis-tagged e.g. fire-doorset paragraphs as "external walls". An untagged chunk
# is better than a wrongly tagged one.
EMBEDDING_THRESHOLD = 0.75  # cosine (bge-small)
EMBEDDING_MARGIN = 0.03  # best node must beat the runner-up by at least this


@dataclass(frozen=True)
class TaxonomyNode:
    """A node in the taxonomy tree."""

    id: str
    label: str
    parent: str | None = None
    description: str = ""
    sections: tuple[str, ...] = ()
    appendix_sections: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass
class Taxonomy:
    """The loaded taxonomy plus its tagging parameters."""

    nodes: dict[str, TaxonomyNode]
    min_keyword_hits: int = 2
    version: str = "draft"
    children: dict[str, list[str]] = field(default_factory=dict)

    @property
    def roots(self) -> list[TaxonomyNode]:
        """Top-level (Requirement) nodes."""
        return [n for n in self.nodes.values() if n.parent is None]

    @property
    def leaves(self) -> list[TaxonomyNode]:
        """Sub-topic nodes."""
        return [n for n in self.nodes.values() if n.parent is not None]

    def label(self, node_id: str) -> str:
        """Display label such as 'B3 › Cavities and cavity barriers'."""
        node = self.nodes[node_id]
        if node.parent:
            return f"{node.parent} › {node.label}"
        return f"{node.id} {node.label}"


def load_taxonomy(path: Path) -> Taxonomy:
    """Load the YAML taxonomy."""
    raw = yaml.safe_load(path.read_text())
    nodes: dict[str, TaxonomyNode] = {}
    children: dict[str, list[str]] = {}
    for top in raw["nodes"]:
        nodes[top["id"]] = TaxonomyNode(top["id"], top["label"])
        children[top["id"]] = []
        for ch in top.get("children", []):
            nodes[ch["id"]] = TaxonomyNode(
                ch["id"],
                ch["label"],
                parent=top["id"],
                description=ch.get("description", ""),
                sections=tuple(ch.get("sections", [])),
                appendix_sections=tuple(ch.get("appendix_sections", [])),
                keywords=tuple(ch.get("keywords", [])),
            )
            children[top["id"]].append(ch["id"])
    return Taxonomy(nodes, raw.get("min_keyword_hits", 2), str(raw.get("version", "draft")), children)


def _contains(haystack: str, needle: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(needle.lower())}(?![A-Za-z])", haystack) is not None


def tag_structure(chunk: Chunk, tax: Taxonomy) -> list[str]:
    """Tags from the document structure: Requirement + section/subsection headings."""
    if chunk.requirement == "Appendix":
        title = chunk.section_title.lower()
        return [n.id for n in tax.leaves if any(_contains(title, s) for s in n.appendix_sections)]
    if chunk.requirement not in tax.children:
        return []
    heading = f"{chunk.section_title} {chunk.subsection or ''}".lower()
    return [
        nid for nid in tax.children[chunk.requirement] if any(_contains(heading, s) for s in tax.nodes[nid].sections)
    ]


def tag_keywords(chunk: Chunk, tax: Taxonomy) -> list[str]:
    """Tags from keyword hits in the chunk text (case-insensitive, whole phrase)."""
    text = chunk.text.lower()
    tags = []
    for node in tax.leaves:
        hits = sum(_contains(text, k) for k in node.keywords)
        if hits >= tax.min_keyword_hits:
            tags.append(node.id)
    return tags


def tag_chunks(
    chunks: list[Chunk],
    tax: Taxonomy,
    embed: Callable[[list[str]], np.ndarray] | None = None,
    chunk_vectors: np.ndarray | None = None,
) -> tuple[list[Chunk], dict[str, dict[str, str]]]:
    """Tag every chunk; returns tagged chunks and ``{chunk_id: {node_id: rule}}``.

    ``embed`` / ``chunk_vectors`` enable the embedding fallback; both must be
    L2-normalised so the dot product is the cosine similarity.
    """
    leaves = tax.leaves
    node_vecs = None
    if embed is not None and chunk_vectors is not None:
        node_vecs = embed([f"{n.label}. {n.description}" for n in leaves])

    out: list[Chunk] = []
    provenance: dict[str, dict[str, str]] = {}
    for i, chunk in enumerate(chunks):
        rules: dict[str, str] = {}
        for nid in tag_structure(chunk, tax):
            rules[nid] = "structure"
        if not rules:
            for nid in tag_keywords(chunk, tax):
                rules.setdefault(nid, "keyword")
        if not rules and node_vecs is not None and chunk_vectors is not None:
            sims = node_vecs @ chunk_vectors[i]
            order = np.argsort(-sims)
            best, second = int(order[0]), int(order[1])
            if sims[best] >= EMBEDDING_THRESHOLD and sims[best] - sims[second] >= EMBEDDING_MARGIN:
                rules[leaves[best].id] = f"embedding ({sims[best]:.2f})"
        for nid in list(rules):
            parent = tax.nodes[nid].parent
            if parent:
                rules.setdefault(parent, "parent")
        if chunk.requirement in tax.children:
            rules.setdefault(chunk.requirement, "structure")
        provenance[chunk.chunk_id] = rules
        out.append(chunk.model_copy(update={"taxonomy": sorted(rules)}))
    return out, provenance


def expand_filter(node_ids: list[str], tax: Taxonomy) -> set[str]:
    """Expand a UI filter selection to include descendants of selected top-level nodes."""
    expanded: set[str] = set()
    for nid in node_ids:
        expanded.add(nid)
        expanded.update(tax.children.get(nid, []))
    return expanded
