"""Append-only audit log and pilot feedback capture (JSON Lines).

Design for data minimisation: no user identifiers, IP addresses or session data are
recorded, and obvious personal data typed into free text (email addresses, phone
numbers, UK postcodes) is redacted before anything is written.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+44\s?|0)(?:\d\s?){9,10}(?!\d)"), "[phone]"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I), "[postcode]"),
]


def redact(text: str) -> str:
    """Redact email addresses, UK phone numbers and postcodes."""
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def now_iso() -> str:
    """UTC timestamp in ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_audit(log_dir: Path, record: dict[str, Any]) -> None:
    """Append one query record to ``audit.jsonl``."""
    record = {**record, "question": redact(record.get("question", ""))}
    _append(log_dir / "audit.jsonl", record)


def write_feedback(log_dir: Path, query_id: str, rating: str, comment: str | None = None) -> dict[str, Any]:
    """Append one pilot-feedback record to ``feedback.jsonl``."""
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")
    record = {
        "timestamp": now_iso(),
        "query_id": query_id,
        "rating": rating,
        "comment": redact(comment.strip())[:1000] if comment else None,
    }
    _append(log_dir / "feedback.jsonl", record)
    return record
