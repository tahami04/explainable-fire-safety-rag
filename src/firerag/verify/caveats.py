"""Deterministic caveats derived from the provenance of cited passages.

These do not depend on the LLM: they are facts about the evidence that a reader
should know before relying on the answer.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from firerag.models import Chunk


def _parse_date(text: str) -> date | None:
    m = re.search(r"(\d{1,2} \w+ \d{4})", text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d %B %Y").date()
    except ValueError:
        return None


def provenance_caveats(cited: list[Chunk], today: date | None = None) -> list[str]:
    """Caveats about validity dates, mixed volumes and table extraction."""
    today = today or date.today()
    out: list[str] = []
    for c in cited:
        if not c.effective_note:
            continue
        when = _parse_date(c.effective_note)
        note = c.effective_note.lower()
        if "from" in note and when and when > today:
            out.append(
                f"{c.citation_label} is amended text marked '{c.effective_note}'; "
                f"it is not yet in force as of {today:%d %B %Y}."
            )
        elif "until" in note and when:
            state = "currently applies" if when >= today else "has been superseded"
            out.append(
                f"{c.citation_label} is marked '{c.effective_note}': it {state} and is "
                "replaced by the 2029 amendment text."
            )
    if len({c.volume for c in cited}) > 1:
        out.append(
            "The answer draws on both Volume 1 (dwellings) and Volume 2 (other buildings); "
            "check which volume applies to your building."
        )
    if any(c.kind == "table" for c in cited):
        out.append(
            "Table text was extracted automatically and its layout may be imperfect; "
            "check the table in the official PDF."
        )
    return list(dict.fromkeys(out))
