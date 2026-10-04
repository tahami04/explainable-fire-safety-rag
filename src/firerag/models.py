"""Core data models shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChunkKind = Literal["paragraph", "table", "diagram", "preamble"]


class Chunk(BaseModel):
    """A citable unit of guidance text with its provenance."""

    chunk_id: str = Field(description="Stable ID, e.g. 'v2-8.12' or 'v2-T2.1'.")
    volume: int = Field(description="1 = Dwellings, 2 = Buildings other than dwellings.")
    requirement: str = Field(description="B1–B5, 'Reg38', 'Appendix' or 'General'.")
    section_title: str
    subsection: str | None = None
    paragraph_ref: str | None = Field(default=None, description="Paragraph number as printed, e.g. '2.5' or 'B6'.")
    kind: ChunkKind = "paragraph"
    page: int | None = Field(default=None, description="Printed page number.")
    pdf_page: int = Field(description="1-based page index in the PDF file.")
    text: str
    effective_note: str | None = Field(
        default=None,
        description="Validity marker printed on the page, e.g. 'Effective until 1 September 2029'.",
    )
    taxonomy: list[str] = Field(default_factory=list)

    @property
    def citation_label(self) -> str:
        """Human-readable citation chip, e.g. '[Vol 2, B3, para 8.12, p. 74]'."""
        parts = [f"Vol {self.volume}", self.requirement]
        if self.paragraph_ref:
            prefix = {"table": "Table", "diagram": "Diagram"}.get(self.kind, "para")
            parts.append(f"{prefix} {self.paragraph_ref}")
        elif self.subsection:
            parts.append(self.subsection)
        parts.append(f"p. {self.page}" if self.page else f"pdf p. {self.pdf_page}")
        return "[" + ", ".join(parts) + "]"

    def context_header(self) -> str:
        """Structural context prepended to the text for indexing and prompting."""
        vol = "Volume 1 (Dwellings)" if self.volume == 1 else "Volume 2 (Buildings other than dwellings)"
        bits = [vol, self.section_title]
        if self.subsection:
            bits.append(self.subsection)
        return " – ".join(bits)

    def index_text(self) -> str:
        """Text used for BM25 and embedding: structural context + body."""
        ref = f"{self.paragraph_ref}: " if self.paragraph_ref else ""
        return f"{self.context_header()}\n{ref}{self.text}"
