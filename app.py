"""Streamlit UI for the explainable fire-safety RAG assistant.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

# Make `src/` importable without `pip install -e .` (also sidesteps macOS marking
# editable-install .pth files as hidden, which Python then skips).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from firerag.config import get_settings  # noqa: E402
from firerag.governance.audit import write_feedback  # noqa: E402
from firerag.pipeline import AskResult, EvidenceItem, Pipeline  # noqa: E402
from firerag.taxonomy.tagger import load_taxonomy  # noqa: E402

st.set_page_config(page_title="Explainable Fire-Safety RAG", page_icon="🧯", layout="wide")

DISCLAIMER = (
    "Demonstration project. Not a substitute for professional fire engineering judgement or the official "
    "guidance. Not affiliated with or endorsed by the UK Government."
)
EXAMPLES = [
    "What is the maximum travel distance in one direction only for an office?",
    "When should a block of flats be fitted with sprinklers?",
    "What size of electrical cable should I use for a cooker circuit?",
]
FAITH_STYLE = {
    "Supported": ("#1b7f3b", "✔ Supported"),
    "Partially supported": ("#a86800", "◐ Partially supported"),
    "Unsupported": ("#b42318", "✖ Unsupported"),
}
CONF_STYLE = {"High": "#1b7f3b", "Medium": "#a86800", "Low": "#b42318", "n/a": "#555"}

st.markdown(
    """
<style>
.chip{display:inline-block;padding:1px 8px;margin:2px 3px;border-radius:10px;font-size:0.78em;
      background:rgba(49,101,196,.12);color:inherit;border:1px solid rgba(49,101,196,.35);white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.75em;color:#fff;margin-left:6px}
.sent{margin:0.35em 0 0.6em 0;line-height:1.55}
.badge{display:inline-block;padding:4px 14px;border-radius:14px;color:#fff;font-weight:600}
mark{background:#ffe58f;color:#000;padding:0 1px;border-radius:2px}
.evtext{white-space:pre-wrap;font-size:0.92em;line-height:1.5}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading index and local models…")
def get_pipeline() -> Pipeline:
    """Load the pipeline once per server process."""
    return Pipeline()


@st.cache_resource
def get_taxonomy():
    """Load the draft taxonomy."""
    return load_taxonomy(get_settings().taxonomy_path)


def highlighted(ev: EvidenceItem) -> str:
    """Evidence text as HTML with matched query terms wrapped in <mark>."""
    text, out, pos = ev.text, [], 0
    for start, end in ev.explanation.get("highlight_spans", []):
        out.append(html.escape(text[pos:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def render_confidence(r: AskResult) -> None:
    """Confidence badge plus the full breakdown of its inputs."""
    c = r.confidence
    colour = CONF_STYLE[c.label]
    overall = "—" if c.overall is None else f"{c.overall:.2f}"
    st.markdown(
        f"<span class='badge' style='background:{colour}'>Confidence: {c.label}</span> &nbsp; overall score {overall}",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric(
        "Retrieval strength", f"{c.retrieval_strength:.2f}", help="Best cross-encoder relevance of the evidence (0–1)."
    )
    cols[1].metric(
        "Faithfulness",
        "—" if c.faithfulness_rate is None else f"{c.faithfulness_rate:.2f}",
        help="Share of answer sentences entailed by their own citations (NLI). Partial = 0.5.",
    )
    cols[2].metric(
        "Self-consistency",
        "—" if c.consistency is None else f"{c.consistency:.2f}",
        help="Agreement in meaning and cited sources across sampled answers.",
    )
    with st.expander("How this confidence was computed"):
        w = c.weights
        st.markdown(
            f"`overall = {w['retrieval']}·retrieval + {w['faithfulness']}·faithfulness + "
            f"{w['consistency']}·consistency` (weights re-normalised over available signals)"
        )
        for reason in c.reasons:
            st.markdown(f"- {reason}")
        if r.consistency:
            k = r.consistency
            st.markdown(
                f"- Samples: {k.n_samples}; mean semantic similarity {k.semantic_similarity:.2f}; "
                f"cited-source overlap (Jaccard) {k.citation_overlap:.2f}; "
                f"abstention agreement {k.abstention_agreement:.2f}."
            )


def render_answer(r: AskResult) -> None:
    """Answer sentences with faithfulness labels and citation chips."""
    for s in r.answer:
        colour, label = FAITH_STYLE[s.faithfulness]
        chips = "".join(
            f"<span class='chip' title='{html.escape(cid)}'>{html.escape(lbl)}</span>"
            for cid, lbl in zip(s.citations, s.citation_labels, strict=True)
        )
        st.markdown(
            f"<div class='sent'>{html.escape(s.sentence)} {chips}"
            f"<span class='pill' style='background:{colour}' title='entailment {s.entailment:.2f}'>"
            f"{label}</span></div>",
            unsafe_allow_html=True,
        )
        notes = []
        if s.faithfulness_note:
            notes.append(s.faithfulness_note)
        if s.invalid_citations:
            notes.append(f"Removed invalid citation(s): {', '.join(s.invalid_citations)}")
        for q in s.quotes:
            mark = "verbatim ✔" if q["verbatim"] else "NOT found verbatim ✖"
            notes.append(f"Quote from {q['chunk_id']} ({mark}): “{q['text']}”")
        if notes:
            st.caption(" · ".join(notes))


def render_evidence(r: AskResult) -> None:
    """Ranked evidence with retrieval explanations and highlighted terms."""
    cited = {c for s in r.answer for c in s.citations}
    for ev in r.evidence:
        e = ev.explanation
        star = "★ cited · " if ev.chunk_id in cited else ""
        rel = e.get("rerank_relevance")
        title = (
            f"#{ev.rank} {star}{ev.citation} — relevance {rel:.2f}"
            if rel is not None
            else f"#{ev.rank} {star}{ev.citation}"
        )
        with st.expander(title, expanded=ev.rank == 1 and not r.answer):
            st.caption(
                f"{ev.section_title}{' › ' + ev.subsection if ev.subsection else ''}  ·  "
                f"chunk `{ev.chunk_id}`  ·  PDF page {ev.pdf_page}"
            )
            if ev.effective_note:
                st.warning(f"Validity marker on this page: {ev.effective_note}")
            st.markdown(f"<div class='evtext'>{highlighted(ev)}</div>", unsafe_allow_html=True)
            st.markdown("**Why this was retrieved**")
            st.table(
                {
                    "signal": ["BM25 (keyword)", "Dense (semantic)", "Reciprocal Rank Fusion", "Cross-encoder rerank"],
                    "score": [
                        f"{e['bm25_score']:.2f}",
                        f"{e['dense_similarity']:.3f}",
                        f"{e['rrf_score']:.4f}",
                        "—" if rel is None else f"{rel:.3f}",
                    ],
                    "rank": [
                        str(e["bm25_rank"] or "not in top 30"),
                        str(e["dense_rank"] or "not in top 30"),
                        str(e["fused_rank"]),
                        str(e["final_rank"]),
                    ],
                }
            )
            terms = ", ".join(e["matched_terms"]) or "none (semantic match only)"
            st.caption(f"Matched query terms: {terms}  ·  Taxonomy: {', '.join(ev.taxonomy) or '—'}")


def render_feedback(r: AskResult) -> None:
    """Thumbs up/down with optional comment, saved to logs/feedback.jsonl."""
    st.markdown("#### Was this useful?")
    key = r.query_id
    if st.session_state.get(f"fb_done_{key}"):
        st.success("Thanks, feedback recorded for pilot review.")
        return
    comment = st.text_input("Optional comment (please don't include personal data)", key=f"fb_c_{key}")
    c1, c2, _ = st.columns([1, 1, 6])
    for col, rating, label in ((c1, "up", "👍 Helpful"), (c2, "down", "👎 Not helpful")):
        if col.button(label, key=f"fb_{rating}_{key}"):
            write_feedback(get_settings().log_dir, r.query_id, rating, comment or None)
            st.session_state[f"fb_done_{key}"] = True
            st.rerun()


def main() -> None:
    """Render the app."""
    s = get_settings()
    tax = get_taxonomy()
    try:
        pipe = get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Settings")
        if pipe.provider is None:
            st.info(
                "**Retrieval-only mode.** No LLM key configured: you will see ranked, explained evidence but "
                "no generated answer. Add `OPENAI_API_KEY` to `.env` for full mode."
            )
        else:
            st.success(f"**Full mode** · {pipe.provider.name} · `{pipe.provider.model}`")
        options = [n.id for n in tax.roots] + [n.id for n in tax.leaves]
        chosen = st.multiselect(
            "Filter by taxonomy (draft)",
            options,
            format_func=tax.label,
            help="Restrict retrieval to passages tagged with these topics.",
        )
        samples = st.slider(
            "Self-consistency samples",
            0,
            5,
            s.consistency_samples,
            help="Extra answers sampled at temperature > 0 to measure agreement. 0 disables.",
            disabled=pipe.provider is None,
        )
        st.divider()
        st.caption(DISCLAIMER)
        st.caption(
            "Source: Approved Document B, Vols 1 & 2 (gov.uk). Contains public sector information "
            "licensed under the Open Government Licence v3.0."
        )

    st.title("🧯 Explainable Fire-Safety Guidance Assistant")
    st.caption(
        "Every claim is traceable to a clause, page and quote; every answer is checked, scored and allowed to refuse."
    )
    st.warning(DISCLAIMER, icon="⚠️")

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES, strict=True):
        if col.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    with st.form("ask"):
        q = st.text_area(
            "Your question", key="q", height=80, placeholder="e.g. When are cavity barriers needed in a dwellinghouse?"
        )
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and q.strip():
        with st.spinner("Retrieving, generating and verifying…"):
            st.session_state["result"] = pipe.ask(q.strip(), taxonomy_filter=chosen, n_samples=samples)

    r: AskResult | None = st.session_state.get("result")
    if r is None:
        return

    st.divider()
    if r.abstained:
        st.error(f"**{r.answer_text}**\n\nWhy: {r.abstention_reason}")
    if r.question_interpretation:
        st.caption(f"Interpreted as: {r.question_interpretation}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        if r.answer:
            st.subheader("Answer")
            render_answer(r)
        elif r.mode == "retrieval_only" and not r.abstained:
            st.info("Retrieval-only mode: the most relevant passages are shown below with explanations.")
        if r.caveats:
            st.subheader("Caveats")
            for c in r.caveats:
                st.markdown(f"- {c}")
        if r.reasoning_pathway:
            with st.expander("Reasoning pathway (evidence-linked reasoning summary)", expanded=True):
                st.caption(
                    "A summary written for the reader that links each step to evidence. It is not the "
                    "model's hidden internal reasoning."
                )
                for i, step in enumerate(r.reasoning_pathway, 1):
                    ev = ", ".join(f"`{e}`" for e in step.evidence) or "_no valid evidence_"
                    st.markdown(f"{i}. {step.step}  \n   ↳ evidence: {ev}")
    with right:
        st.subheader("Confidence")
        render_confidence(r)

    st.subheader("Evidence")
    render_evidence(r)
    st.divider()
    render_feedback(r)
    st.caption(
        f"Query ID `{r.query_id}` · logged to `logs/audit.jsonl` · prompt `{r.prompt_version}` · "
        f"latency {r.latency_ms.get('total', 0) / 1000:.1f}s"
    )


main()
