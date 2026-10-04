from __future__ import annotations

from firerag.generate.schema import AnswerSentence, GroundedAnswer, Quote
from firerag.verify.citations import normalise_for_match, validate_citations
from firerag.verify.faithfulness import (
    SentenceFaithfulness,
    check_sentence,
    faithfulness_rate,
    numbers_grounded,
    table_fallback,
)

from .conftest import entail_all, entail_none

EVIDENCE = {
    "v2-8.6": "Stairs and service shafts connecting compartments should be protected to restrict the spread of fire "
    "between the compartments. These are called ‘protected shafts’.",
    "v2-T2.1": "Table 2.1 Limitations on travel distance\n3 | Office | 18 | 45",
}


def _answer(*sentences: AnswerSentence) -> GroundedAnswer:
    return GroundedAnswer(question_interpretation="q", answer=list(sentences))


def test_invalid_citation_ids_are_removed():
    ans = _answer(AnswerSentence(sentence="Shafts are protected.", citations=["v2-8.6", "v9-made-up"]))
    [rep] = validate_citations(ans, EVIDENCE)
    assert rep.valid == ["v2-8.6"]
    assert rep.invalid == ["v9-made-up"]


def test_sentence_with_only_invalid_citations_has_no_valid_citation():
    [rep] = validate_citations(_answer(AnswerSentence(sentence="x", citations=["nope"])), EVIDENCE)
    assert not rep.has_valid_citation


def test_verbatim_quote_accepts_typography_and_whitespace_differences():
    q = Quote(chunk_id="v2-8.6", text="These are called 'protected   shafts'.")
    [rep] = validate_citations(_answer(AnswerSentence(sentence="s", citations=["v2-8.6"], quotes=[q])), EVIDENCE)
    assert rep.all_quotes_verbatim


def test_paraphrased_quote_is_flagged():
    q = Quote(chunk_id="v2-8.6", text="Shafts must always be protected from fire.")
    [rep] = validate_citations(_answer(AnswerSentence(sentence="s", citations=["v2-8.6"], quotes=[q])), EVIDENCE)
    assert not rep.all_quotes_verbatim


def test_quote_attributed_to_wrong_chunk_is_flagged():
    q = Quote(chunk_id="v2-T2.1", text="These are called protected shafts")
    [rep] = validate_citations(_answer(AnswerSentence(sentence="s", citations=["v2-T2.1"], quotes=[q])), EVIDENCE)
    assert not rep.all_quotes_verbatim


def test_normalise_keeps_numbers_and_words():
    assert normalise_for_match("18 m") != normalise_for_match("45 m")


def test_faithfulness_labels():
    sup = check_sentence("Shafts are protected.", ["v2-8.6"], EVIDENCE, entail_all, 0.5, 0.15)
    uns = check_sentence("Shafts are blue.", ["v2-8.6"], EVIDENCE, entail_none, 0.5, 0.15)
    none = check_sentence("No citation.", [], EVIDENCE, entail_all, 0.5, 0.15)
    assert sup.label == "Supported" and uns.label == "Unsupported" and none.label == "Unsupported"
    assert faithfulness_rate(["Supported", "Partially supported", "Unsupported"]) == 0.5


def test_table_fallback_requires_grounded_numbers_and_verbatim_quote():
    base = SentenceFaithfulness(
        label="Unsupported", entailment=0.01, contradiction=0.0, best_chunk="v2-T2.1", per_citation={}
    )
    tables = {"v2-T2.1": EVIDENCE["v2-T2.1"]}
    ok = table_fallback(base, "Offices allow 18m in one direction.", tables, quotes_verbatim=True, has_quote=True)
    wrong_number = table_fallback(base, "Offices allow 25m.", tables, quotes_verbatim=True, has_quote=True)
    no_quote = table_fallback(base, "Offices allow 18m.", tables, quotes_verbatim=True, has_quote=False)
    assert ok.label == "Partially supported" and "Table evidence" in ok.note
    assert wrong_number.label == "Unsupported"
    assert no_quote.label == "Unsupported"
    assert numbers_grounded("18 and 45", EVIDENCE["v2-T2.1"])


def test_hypothesis_cleaning_strips_inline_citations_and_quotes():
    from firerag.verify.faithfulness import clean_hypothesis

    s = 'Shafts should be protected. [v2-8.6] "These are called protected shafts".'
    assert clean_hypothesis(s) == "Shafts should be protected."


def test_premise_prefix_is_passed_to_scorer():
    seen = []

    def spy(pairs):
        seen.extend(pairs)
        return entail_all(pairs)

    check_sentence("x", ["v2-8.6"], EVIDENCE, spy, 0.5, 0.15, premise_prefix={"v2-8.6": "In Volume 2: "})
    assert all(p.startswith("In Volume 2: ") for p, _ in seen)
