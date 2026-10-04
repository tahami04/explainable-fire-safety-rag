"""Chunk metadata tests on a synthetic PDF that mimics Approved Document B's layout."""

from __future__ import annotations

import pymupdf
import pytest

from firerag.ingest.chunker import chunk_volume, repair_ligatures
from firerag.ingest.textclean import apply_repairs, build_repair_map, build_vocab


def _page(doc, printed: int, lines: list[tuple[str, float, bool]]) -> None:
    page = doc.new_page(width=595, height=842)
    y = 100.0
    for text, size, bold in lines:
        page.insert_text((57, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size + 10
    page.insert_text((57, 815), f"{printed}  Approved Document B Volume 2, 2019 edition", fontsize=9)
    page.insert_text((200, 30), "O N L I N E V E R S I O N", fontsize=9)  # running header


@pytest.fixture
def synthetic_pdf(tmp_path):
    doc = pymupdf.open()
    _page(
        doc, 1, [("Introduction", 26, True), ("This document gives guidance on fire safety in buildings.", 12, False)]
    )
    _page(
        doc,
        12,
        [
            ("Requirement B1: Means of warning and escape", 26, True),
            ("The building shall be designed so that there are appropriate means of escape.", 12, False),
        ],
    )
    _page(
        doc,
        13,
        [
            ("Section 2: Design for horizontal escape", 26, True),
            ("Escape route design", 20, True),
            ("2.1 Means of escape should be provided from any point on a storey.", 12, False),
            ("2.5m wide corridors are mentioned here but this is not a paragraph.", 12, False),
            ("2.2 For small premises the guidance in section 4 may be followed.", 12, False),
            ("2.3Fire and smoke dampers should meet both of the following conditions.", 12, False),
            ("Table 2.1 Limitations on travel distance", 12, True),
            ("Office   18   45", 12, False),
        ],
    )
    _page(doc, 14, [("Index", 26, True), ("Alarm 1.1", 12, False)])
    doc.set_toc(
        [
            [1, "Introduction", 1],
            [1, "Requirement B1: Means of warning and escape", 2],
            [1, "Section 2: Design for horizontal escape", 3],
            [1, "Index", 4],
        ]
    )
    path = tmp_path / "synthetic.pdf"
    doc.save(path)
    return path


def test_chunks_have_required_metadata(synthetic_pdf):
    chunks = chunk_volume(synthetic_pdf, volume=2)
    by_id = {c.chunk_id: c for c in chunks}
    assert {"v2-2.1", "v2-2.2", "v2-2.3", "v2-T2.1"} <= set(by_id)
    c = by_id["v2-2.1"]
    assert c.volume == 2
    assert c.requirement == "B1"
    assert c.section_title == "Section 2: Design for horizontal escape"
    assert c.subsection == "Escape route design"
    assert c.paragraph_ref == "2.1"
    assert c.page == 13 and c.pdf_page == 3
    assert c.text and "O N L I N E" not in c.text
    assert c.citation_label == "[Vol 2, B1, para 2.1, p. 13]"


def test_dimension_is_not_mistaken_for_paragraph(synthetic_pdf):
    chunks = chunk_volume(synthetic_pdf, volume=2)
    assert "v2-2.5" not in {c.chunk_id for c in chunks}
    by_id = {c.chunk_id: c for c in chunks}
    assert "2.5m wide corridors" in by_id["v2-2.1"].text


def test_paragraph_number_without_space_is_detected(synthetic_pdf):
    by_id = {c.chunk_id: c for c in chunk_volume(synthetic_pdf, volume=2)}
    assert by_id["v2-2.3"].text.startswith("Fire and smoke dampers")


def test_table_chunk_and_index_pages_excluded(synthetic_pdf):
    chunks = chunk_volume(synthetic_pdf, volume=2)
    table = next(c for c in chunks if c.chunk_id == "v2-T2.1")
    assert table.kind == "table" and "Office" in table.text
    assert not any("Alarm 1.1" in c.text for c in chunks)


def test_chunk_ids_unique(synthetic_pdf):
    chunks, _ = repair_ligatures(chunk_volume(synthetic_pdf, volume=2))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ligature_repair_is_corpus_driven():
    texts = ["fire " * 10 + "fre " * 3 + "free soft flat final"] + ["first floor fire"] * 3
    repairs = build_repair_map(build_vocab(texts))
    assert repairs["fre"] == "fire"
    assert "free" not in repairs and "soft" not in repairs
    assert apply_repairs("Fre doors", repairs) == "Fire doors"
