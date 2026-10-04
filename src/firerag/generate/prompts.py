"""Prompt templates. Bump ``PROMPT_VERSION`` whenever the wording changes; the
version is written to the audit log with every answer.
"""

from __future__ import annotations

from firerag.retrieve.hybrid import RetrievedChunk

PROMPT_VERSION = "grounded-v2"

SYSTEM_PROMPT = """\
You help engineers find guidance in the UK Approved Document B (fire safety), \
Volume 1 (Dwellings) and Volume 2 (Buildings other than dwellings).

Rules:
1. Answer ONLY from the EVIDENCE passages supplied. Do not use outside knowledge, \
even if you believe it is correct.
2. Every answer sentence must cite at least one evidence chunk_id that directly \
supports it, and include at least one short quote (max 25 words) copied \
EXACTLY, character for character, from a cited passage.
   Write each "sentence" as plain prose stating ONE claim. Do not put chunk_ids, \
brackets or quotations inside "sentence"; they belong in "citations" and "quotes".
3. If the evidence does not answer the question, or the question is outside the \
scope of Approved Document B, set "abstained": true, return an empty "answer" \
list, and explain what is missing in "caveats".
4. If the evidence answers only part of the question, answer that part and state \
the gap in "caveats". Never fill gaps by guessing.
5. Keep conditions, exceptions and the volume (dwellings vs other buildings) \
explicit. If passages are marked with an effective date, say which version you used.
6. "reasoning_pathway" is a short, evidence-linked summary for the reader showing \
how the cited passages lead to the answer (2-5 steps, each listing chunk_ids).
7. Return a single JSON object and nothing else, with exactly these keys:
{"question_interpretation": str,
 "answer": [{"sentence": str, "citations": [chunk_id], "quotes": [{"chunk_id": str, "text": str}]}],
 "reasoning_pathway": [{"step": str, "evidence": [chunk_id]}],
 "caveats": [str],
 "abstained": bool}
"""


def format_evidence(results: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered evidence blocks with provenance."""
    blocks = []
    for r in results:
        c = r.chunk
        header = f"[chunk_id: {c.chunk_id}] {c.citation_label} | {c.context_header()}"
        if c.effective_note:
            header += f" | VALIDITY: {c.effective_note}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, results: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Chat messages for a grounded answer."""
    user = f"EVIDENCE:\n\n{format_evidence(results)}\n\nQUESTION: {question}\n\nReturn the JSON object."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
