"""Parse Approved Document B PDFs into a stream of body-text lines with provenance.

We use PyMuPDF's layout information (block coordinates, font flags) to separate
running headers/footers from body text and to recognise headings, and the PDF's
own table of contents to know which Requirement and Section each page belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

HEADER_MAX_Y = 70.0  # running header, requirement tab, "ONLINE VERSION" banner
FOOTER_MIN_Y = 795.0  # printed page number and edition footer
# Collated editions print both current and future-amended text, marked e.g.
# "Effective until 1 September 2029" / "2029 amendment – effective from 2 September 2029".
_EFFECTIVE = re.compile(r"(?:\d{4} amendment – )?[Ee]ffective (?:until|from) \d{1,2} \w+ \d{4}")
_NOISE = re.compile(
    r"^(O N L I N E V E R S I O N|Building Regulations 2010|"
    r"Approved Document B Volume \d, 2019 edition)$"
)


@dataclass(frozen=True)
class Line:
    """One line of body text with layout hints."""

    text: str
    pdf_page: int  # 1-based
    printed_page: int | None
    is_bold: bool
    font_size: float
    effective_note: str | None
    y0: float = 0.0
    y1: float = 0.0


@dataclass(frozen=True)
class TocEntry:
    """A level-1 or level-2 table-of-contents entry."""

    level: int
    title: str
    pdf_page: int  # 1-based


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").replace("‑", "-")).strip()


def read_toc(doc: pymupdf.Document) -> list[TocEntry]:
    """Return cleaned level-1/2 TOC entries."""
    return [TocEntry(lvl, _clean(title), page) for lvl, title, page in doc.get_toc() if lvl <= 2]


def _page_meta(page: pymupdf.Page) -> tuple[int | None, str | None]:
    """Find the printed page number and any 'Effective until/from' marker on a page."""
    printed, effective = None, None
    for _x0, y0, _x1, _y1, text, *_ in page.get_text("blocks"):
        flat = _clean(text)
        m = _EFFECTIVE.search(flat)
        if m:
            effective = m.group(0)
        if y0 >= FOOTER_MIN_Y - 10:
            nums = re.findall(r"(?:^|\s)(\d{1,3})(?:\s|$)", flat)
            if nums and printed is None:
                printed = int(nums[0])
    return printed, effective


def iter_lines(pdf_path: Path, first_pdf_page: int, last_pdf_page: int) -> list[Line]:
    """Extract body lines from ``first_pdf_page``..``last_pdf_page`` (1-based, inclusive)."""
    doc = pymupdf.open(pdf_path)
    out: list[Line] = []
    for pno in range(first_pdf_page - 1, last_pdf_page):
        page = doc[pno]
        printed, effective = _page_meta(page)
        data = page.get_text("dict")
        for block in data["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y1 <= HEADER_MAX_Y or y0 >= FOOTER_MIN_Y:
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in line["spans"]))
                if not text or _NOISE.match(text) or _EFFECTIVE.fullmatch(text.rstrip(" –-")):
                    continue
                bold = all(("Bold" in s["font"]) or (s["flags"] & 16) for s in spans)
                size = max(s["size"] for s in spans)
                out.append(Line(text, pno + 1, printed, bold, size, effective, y0, y1))
    return out


def table_rows(
    page: pymupdf.Page, y_top: float, y_bottom: float, row_tol: float = 3.0, cell_gap: float = 8.0
) -> list[str]:
    """Rebuild a borderless table as ``cell | cell | cell`` rows.

    Words are grouped into visual rows by vertical centre, then split into cells
    wherever the horizontal gap between consecutive words exceeds ``cell_gap``.
    """
    words = [
        w
        for w in page.get_text("words")
        if y_top - 1 <= w[1] and w[3] <= y_bottom + 1 and HEADER_MAX_Y < w[1] < FOOTER_MIN_Y
    ]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[tuple[float, list[tuple]]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) <= row_tol:
            rows[-1][1].append(w)
        else:
            rows.append((yc, [w]))
    out: list[str] = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[0])
        cells: list[list[tuple]] = [[ws[0]]]
        for a, b in zip(ws, ws[1:], strict=False):
            if b[0] - a[2] < cell_gap:
                cells[-1].append(b)
            else:
                cells.append([b])
        out.append(" | ".join(_clean(" ".join(x[4] for x in c)) for c in cells))
    return out
