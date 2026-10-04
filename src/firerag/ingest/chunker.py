"""Structure-aware chunking of Approved Document B.

Chunk boundaries follow the document's own structure, in priority order:

1. Requirement (B1–B5) / Section / Appendix headings (26pt bold).
2. Numbered paragraphs (``8.12`` in sections, ``B6`` in appendices). A paragraph
   number only counts if it belongs to the current section (``8.x`` inside
   Section 8) and increases monotonically, which rejects dimensions such as
   "2.5m" that happen to start a line.
3. Table and diagram captions ("Table 2.1 Limitations on travel distance"). A line
   only counts as a caption when the label is followed by nothing or by a
   capitalised title, so body sentences like "Table 2.2 gives…" are not captions.
   Table text is rebuilt row by row from word coordinates (see ``table_rows``).

Paragraphs longer than ``MAX_WORDS`` fall back to overlapping word windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from firerag.ingest.pdf_parser import Line, iter_lines, read_toc, table_rows
from firerag.ingest.textclean import apply_repairs, build_repair_map, build_vocab, normalise_whitespace
from firerag.models import Chunk, ChunkKind

MAX_WORDS = 380  # ≈ 500 tokens
WINDOW_WORDS = 320  # ≈ 420 tokens
OVERLAP_WORDS = 60
MIN_PREAMBLE_WORDS = 8

_SECTION = re.compile(r"^Section (\d{1,2})\s*:")
_REQUIREMENT = re.compile(r"^Requirement (B[1-5])\s*:")
_APPENDIX = re.compile(r"^Appendix ([A-G])\s*:")
_REGULATION = re.compile(r"^Regulation (\d+)\s*:")
_PARA_NUM = re.compile(r"^(\d{1,2})\.(\d{1,3})(?:\s+|$|(?=[A-Z‘]))(.*)$")
_APP_NUM = re.compile(r"^([A-G])(\d{1,2})(?:\s+|$|(?=[A-Z‘]))(.*)$")
_CAPTION = re.compile(r"^(Table|Diagram)\s+([A-G]?\d{1,2}(?:\.\d{1,2})?[a-z]?)\b\s*(.*)$")
_LIST_ITEM = re.compile(r"^(?:[a-z]|[ivx]{1,4})\.(?:\s|$)")


@dataclass
class _Draft:
    """A chunk being accumulated."""

    kind: ChunkKind
    ref: str | None
    requirement: str
    section_title: str
    subsection: str | None
    lines: list[Line] = field(default_factory=list)


@dataclass
class _State:
    requirement: str = "General"
    section_title: str = "Introduction"
    section_num: int | None = None
    appendix: str | None = None
    sub2: str | None = None
    sub3: str | None = None
    last_minor: int = 0

    @property
    def subsection(self) -> str | None:
        parts = [p for p in (self.sub2, self.sub3) if p]
        return " › ".join(parts) or None


def _join(lines: list[Line], kind: ChunkKind) -> str:
    """Join lines into readable text, keeping list items on their own line."""
    if kind in ("table", "diagram"):
        return normalise_whitespace("\n".join(ln.text for ln in lines))
    out: list[str] = []
    for ln in lines:
        if not out or _LIST_ITEM.match(ln.text):
            out.append(ln.text)
        else:
            out[-1] = out[-1] + " " + ln.text
    return normalise_whitespace("\n".join(out))


def _windows(words: list[str]) -> list[str]:
    """Split a long word list into overlapping windows."""
    step = WINDOW_WORDS - OVERLAP_WORDS
    return [" ".join(words[i : i + WINDOW_WORDS]) for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]


def chunk_volume(pdf_path: Path, volume: int) -> list[Chunk]:
    """Parse one volume into chunks (before ligature repair and tagging)."""
    doc = pymupdf.open(pdf_path)
    toc = read_toc(doc)
    start = next(e.pdf_page for e in toc if e.level == 1 and e.title.startswith("Introduction"))
    end = next(e.pdf_page for e in toc if e.level == 1 and e.title.startswith("Index")) - 1
    lines = iter_lines(pdf_path, start, end)

    st = _State()
    drafts: list[_Draft] = []
    current: _Draft | None = None
    pending_heading: list[str] = []

    def new_draft(kind: ChunkKind, ref: str | None) -> _Draft:
        d = _Draft(kind, ref, st.requirement, st.section_title, st.subsection)
        drafts.append(d)
        return d

    def flush_heading() -> None:
        """Apply an accumulated (possibly two-line) level-1 heading."""
        nonlocal current
        if not pending_heading:
            return
        title = " ".join(pending_heading)
        pending_heading.clear()
        st.section_title, st.sub2, st.sub3, st.last_minor = title, None, None, 0
        if m := _REQUIREMENT.match(title):
            st.requirement, st.section_num, st.appendix = m.group(1), None, None
        elif m := _SECTION.match(title):
            st.section_num, st.appendix = int(m.group(1)), None
            if st.section_num == 0:
                st.requirement = "General"
        elif m := _APPENDIX.match(title):
            st.requirement, st.section_num, st.appendix = "Appendix", None, m.group(1)
        elif _REGULATION.match(title):
            st.requirement, st.section_num, st.appendix = "Reg38", None, None
        current = None

    prev_heading: tuple[str, float, int] | None = None  # (level, size, pdf_page)
    for ln in lines:
        text = ln.text
        if ln.is_bold and ln.font_size >= 24:
            pending_heading.append(text)
            continue
        flush_heading()

        is_sub2 = ln.is_bold and 18 <= ln.font_size < 24
        is_sub3 = ln.is_bold and 13.5 <= ln.font_size < 18 and not _CAPTION.match(text)
        if (is_sub2 or is_sub3) and prev_heading == ("sub2" if is_sub2 else "sub3", ln.font_size, ln.pdf_page):
            # Heading wrapped onto a second line: append to the one just set.
            if is_sub2:
                st.sub2 = f"{st.sub2} {text}"
            else:
                st.sub3 = f"{st.sub3} {text}"
            continue
        if is_sub2:
            st.sub2, st.sub3 = text, None
            current = None
            prev_heading = ("sub2", ln.font_size, ln.pdf_page)
            continue
        if is_sub3:
            st.sub3 = text
            if current is not None and current.kind in ("table", "diagram"):
                current = None
            prev_heading = ("sub3", ln.font_size, ln.pdf_page)
            continue
        prev_heading = None

        cap = _CAPTION.match(text)
        if cap and cap.group(3) and not (cap.group(3)[0].isupper() or cap.group(3).startswith("(cont")):
            cap = None
        if cap:
            kind: ChunkKind = "table" if cap.group(1) == "Table" else "diagram"
            current = new_draft(kind, cap.group(2))
            current.lines.append(ln)
            continue

        para = _PARA_NUM.match(text)
        if para and st.section_num is not None and int(para.group(1)) == st.section_num:
            minor = int(para.group(2))
            if minor > st.last_minor:
                st.last_minor = minor
                current = new_draft("paragraph", f"{para.group(1)}.{para.group(2)}")
                if para.group(3):
                    current.lines.append(
                        Line(para.group(3), ln.pdf_page, ln.printed_page, ln.is_bold, ln.font_size, ln.effective_note)
                    )
                continue
        app = _APP_NUM.match(text)
        if app and st.appendix is not None and app.group(1) == st.appendix:
            minor = int(app.group(2))
            if minor > st.last_minor:
                st.last_minor = minor
                current = new_draft("paragraph", f"{app.group(1)}{app.group(2)}")
                if app.group(3):
                    current.lines.append(
                        Line(app.group(3), ln.pdf_page, ln.printed_page, ln.is_bold, ln.font_size, ln.effective_note)
                    )
                continue

        if current is None:
            current = new_draft("preamble", None)
        current.lines.append(ln)
    flush_heading()

    return _finalise(drafts, volume, doc)


def _table_text(doc: pymupdf.Document, lines: list[Line]) -> str:
    """Rebuild a table chunk's text row by row from the page regions it covers."""
    rows: list[str] = []
    for pno in sorted({ln.pdf_page for ln in lines}):
        on_page = [ln for ln in lines if ln.pdf_page == pno]
        top = min(ln.y0 for ln in on_page)
        bottom = max(ln.y1 for ln in on_page)
        rows.extend(table_rows(doc[pno - 1], top, bottom))
    return normalise_whitespace("\n".join(rows))


def _finalise(drafts: list[_Draft], volume: int, doc: pymupdf.Document) -> list[Chunk]:
    """Turn drafts into Chunk objects with stable, unique IDs."""
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for d in drafts:
        if not d.lines or d.section_title.lower().startswith("contents"):
            continue
        text = _table_text(doc, d.lines) if d.kind == "table" else _join(d.lines, d.kind)
        words = text.split()
        if d.kind == "preamble" and len(words) < MIN_PREAMBLE_WORDS:
            continue
        if d.kind == "paragraph":
            base = f"v{volume}-{d.ref}"
        elif d.kind in ("table", "diagram"):
            base = f"v{volume}-{'T' if d.kind == 'table' else 'D'}{d.ref}"
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", d.section_title.lower()).strip("-")[:24]
            base = f"v{volume}-pre-{slug}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        cid = base if n == 0 else f"{base}~{n + 1}"

        pages = [ln.printed_page for ln in d.lines if ln.printed_page is not None]
        notes = {ln.effective_note for ln in d.lines if ln.effective_note}
        common = dict(
            volume=volume,
            requirement=d.requirement,
            section_title=d.section_title,
            subsection=d.subsection,
            paragraph_ref=d.ref,
            kind=d.kind,
            page=pages[0] if pages else None,
            pdf_page=d.lines[0].pdf_page,
            effective_note=sorted(notes)[0] if notes else None,
        )
        if len(words) <= MAX_WORDS or d.kind in ("table", "diagram") and len(words) <= 2 * MAX_WORDS:
            chunks.append(Chunk(chunk_id=cid, text=text, **common))
        else:
            for i, win in enumerate(_windows(words), start=1):
                chunks.append(Chunk(chunk_id=f"{cid}-w{i}", text=win, **common))
    return chunks


def repair_ligatures(chunks: list[Chunk]) -> tuple[list[Chunk], dict[str, str]]:
    """Repair dropped-ligature words across all chunks using a corpus-wide vocabulary."""
    vocab = build_vocab(c.text for c in chunks)
    repairs = build_repair_map(vocab)
    fixed = [
        c.model_copy(
            update={
                "text": apply_repairs(c.text, repairs),
                "section_title": apply_repairs(c.section_title, repairs),
                "subsection": apply_repairs(c.subsection, repairs) if c.subsection else None,
            }
        )
        for c in chunks
    ]
    return fixed, repairs
