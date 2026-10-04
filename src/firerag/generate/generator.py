"""Grounded generation: one deterministic answer plus N sampled answers for the
self-consistency check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from firerag.generate.prompts import build_messages
from firerag.generate.providers import LLMProvider
from firerag.generate.schema import GroundedAnswer
from firerag.retrieve.hybrid import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class Generation:
    """The primary answer, the consistency samples and any parse failures."""

    primary: GroundedAnswer | None
    samples: list[GroundedAnswer] = field(default_factory=list)
    parse_failures: int = 0
    raw_primary: str = ""


def parse_answer(raw: str) -> GroundedAnswer | None:
    """Parse and validate a JSON answer; returns ``None`` if it is malformed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return GroundedAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        log.warning("Could not parse LLM answer: %s", exc)
        return None


def generate(
    provider: LLMProvider, question: str, evidence: list[RetrievedChunk], n_samples: int, sample_temperature: float
) -> Generation:
    """Produce the primary (temperature 0) answer and ``n_samples`` sampled answers."""
    messages = build_messages(question, evidence)
    failures = 0

    raw = provider.complete(messages, temperature=0.0, n=1)[0]
    primary = parse_answer(raw)
    if primary is None:  # one retry for malformed JSON
        failures += 1
        raw = provider.complete(messages, temperature=0.0, n=1)[0]
        primary = parse_answer(raw)
        failures += primary is None

    samples: list[GroundedAnswer] = []
    if n_samples > 0 and primary is not None:
        for s in provider.complete(messages, temperature=sample_temperature, n=n_samples):
            parsed = parse_answer(s)
            if parsed is None:
                failures += 1
            else:
                samples.append(parsed)
    return Generation(primary, samples, failures, raw)
