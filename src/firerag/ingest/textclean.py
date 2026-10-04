"""Text clean-up for PDF extraction artefacts.

The gov.uk PDFs embed some fonts whose ``fi``/``ff``/``fl`` ligature glyphs have no
Unicode mapping, so PyMuPDF returns e.g. ``fre`` for "fire" and ``efect`` for
"effect" on roughly a quarter of pages. Left alone this silently breaks keyword
retrieval, so we repair it with a corpus-driven rule rather than a hand-made list.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Iterable

_WORD = re.compile(r"[A-Za-z]+")
_LIGATURES = ("fi", "ff", "fl", "ffi")
_MAX_REPAIRS_PER_WORD = 2


def build_vocab(texts: Iterable[str]) -> Counter[str]:
    """Count lower-cased word frequencies across the whole corpus."""
    vocab: Counter[str] = Counter()
    for text in texts:
        vocab.update(w.lower() for w in _WORD.findall(text))
    return vocab


def _candidates(word: str) -> Iterable[str]:
    """Yield variants of ``word`` with one or two ``f`` replaced by a ligature."""
    positions = [i for i, ch in enumerate(word) if ch in "fF"]
    for k in range(1, min(_MAX_REPAIRS_PER_WORD, len(positions)) + 1):
        for combo in itertools.combinations(positions, k):
            for ligs in itertools.product(_LIGATURES, repeat=k):
                chars = list(word)
                for pos, lig in zip(combo, ligs, strict=True):
                    chars[pos] = chars[pos] + lig[1:]
                yield "".join(chars)


def build_repair_map(vocab: Counter[str]) -> dict[str, str]:
    """Map broken words to their repaired form.

    A word is repaired only when a ligature variant exists in the same corpus and
    is *more frequent* than the broken form. "fre" (≈250 occurrences) becomes
    "fire" (≈2,000); genuine words such as "free" have no more-frequent variant and
    are left untouched.
    """
    repairs: dict[str, str] = {}
    for word, count in vocab.items():
        if "f" not in word:
            continue
        best, best_count = None, count
        for cand in _candidates(word):
            c = vocab.get(cand, 0)
            if c > best_count:
                best, best_count = cand, c
        if best is not None:
            repairs[word] = best
    return repairs


def apply_repairs(text: str, repairs: dict[str, str]) -> str:
    """Apply ``repairs`` to ``text`` preserving the case of the first letter."""
    if not repairs:
        return text

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        repl = repairs.get(word.lower())
        if repl is None:
            return word
        if word[0].isupper():
            return repl[0].upper() + repl[1:]
        return repl

    return _WORD.sub(fix, text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and normalise non-breaking spaces."""
    text = text.replace("\xa0", " ").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
