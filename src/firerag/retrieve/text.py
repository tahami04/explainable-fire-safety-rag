"""Tokenisation and query-term highlighting shared by BM25 and the explanation layer."""

from __future__ import annotations

import re

# Paragraph refs such as "8.12" and classes such as "a2-s1" survive as single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from has have how i if in
    into is it its may might must my no not of on or our should so such than that the
    their them then there these they this to under up was we were what when where which
    while who why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed and a light plural strip."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def matched_terms(query: str, text: str) -> list[str]:
    """Query tokens that also occur in ``text`` (order of first appearance in the query)."""
    doc = set(tokenize(text))
    seen: list[str] = []
    for tok in tokenize(query):
        if tok in doc and tok not in seen:
            seen.append(tok)
    return seen


def highlight_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    """Character spans in ``text`` of words whose token form is in ``terms``."""
    if not terms:
        return []
    wanted = set(terms)
    spans = []
    for m in _TOKEN.finditer(text.lower()):
        toks = tokenize(m.group(0))
        if toks and toks[0] in wanted:
            spans.append((m.start(), m.end()))
    return spans
