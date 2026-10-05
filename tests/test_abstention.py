from __future__ import annotations

from datetime import date

from firerag.config import ABSTAIN_MESSAGE
from firerag.pipeline import Pipeline
from firerag.verify.caveats import provenance_caveats
from firerag.verify.confidence import post_generation_abstention, retrieval_gate, score_confidence

from .conftest import CHUNKS, FakeLLM, entail_all, entail_none, fake_embed

GOOD = {
    "question_interpretation": "Protection of shafts between compartments.",
    "answer": [
        {
            "sentence": "Stairs and service shafts connecting compartments should be protected.",
            "citations": ["v2-8.6"],
            "quotes": [{"chunk_id": "v2-8.6", "text": "These are called protected shafts"}],
        }
    ],
    "reasoning_pathway": [{"step": "Para 8.6 defines protected shafts.", "evidence": ["v2-8.6", "bogus"]}],
    "caveats": [],
    "abstained": False,
}


def _pipeline(settings, retriever, llm, nli=entail_all):
    return Pipeline(settings, retriever=retriever, provider=llm, nli_scorer=nli, embed=fake_embed)


def test_retrieval_gate(settings):
    assert retrieval_gate(0.01, settings).abstain
    assert not retrieval_gate(0.9, settings).abstain


def test_post_generation_rules(settings):
    assert post_generation_abstention(True, 3, 0.9, settings).abstain
    assert post_generation_abstention(False, 0, 0.9, settings).abstain
    assert post_generation_abstention(False, 2, 0.1, settings).abstain
    assert post_generation_abstention(False, 2, 0.9, settings, faithfulness=0.0).abstain
    assert not post_generation_abstention(False, 2, 0.9, settings, faithfulness=1.0).abstain


def test_confidence_labels_and_caps(settings):
    assert score_confidence(0.95, 1.0, 0.9, 0, settings).label == "High"
    assert score_confidence(0.95, 1.0, 0.9, 1, settings).label == "Medium"  # unsupported sentence caps High
    assert score_confidence(1.0, 0.0, 1.0, 2, settings).label == "Low"  # faithfulness cap
    rel_only = score_confidence(0.9, 1.0, None, 0, settings)
    assert rel_only.consistency is None and rel_only.overall is not None


def test_full_answer_is_verified(settings, retriever):
    llm = FakeLLM(GOOD)
    r = _pipeline(settings, retriever, llm).ask("Do stair shafts between compartments need protection?")
    assert not r.abstained
    assert r.answer[0].faithfulness == "Supported"
    assert r.answer[0].citation_labels == ["[Vol 2, B3, para 8.6, p. 65]"]
    assert r.reasoning_pathway[0].evidence == ["v2-8.6"]  # unknown evidence id dropped
    assert r.consistency is not None and r.consistency.score == 1.0
    assert llm.calls[0] == (0.0, 1) and llm.calls[1][1] == settings.consistency_samples
    assert (settings.log_dir / "audit.jsonl").exists()


def test_model_abstention_returns_standard_message(settings, retriever):
    payload = {**GOOD, "answer": [], "abstained": True, "caveats": ["Not covered."]}
    r = _pipeline(settings, retriever, FakeLLM(payload)).ask("What about shafts in compartments?")
    assert r.abstained and r.answer_text == ABSTAIN_MESSAGE


def test_unsupported_answer_is_withheld(settings, retriever):
    r = _pipeline(settings, retriever, FakeLLM(GOOD), nli=entail_none).ask("Do compartment shafts need protection?")
    assert r.abstained and r.answer == []
    assert "supported" in (r.abstention_reason or "")


def test_out_of_scope_question_abstains_before_generation(settings, retriever):
    settings.min_retrieval_strength = 0.99
    llm = FakeLLM(GOOD)
    r = _pipeline(settings, retriever, llm).ask("Best chocolate cake recipe?")
    assert r.abstained and llm.calls == []


def test_retrieval_only_mode_without_provider(settings, retriever):
    r = Pipeline(settings, retriever=retriever, provider=None, embed=fake_embed).ask("protected shafts")
    assert r.mode == "retrieval_only" and r.answer == [] and r.evidence
    assert r.evidence[0].explanation["matched_terms"]


def test_future_amendment_caveat():
    caveats = provenance_caveats([CHUNKS[2]], today=date(2026, 10, 5))
    assert any("not yet in force" in c for c in caveats)


def test_high_requires_strong_faithfulness(settings):
    assert score_confidence(1.0, 0.5, 1.0, 0, settings).label == "Medium"


def test_abstention_has_no_confidence_label(settings, retriever):
    payload = {**GOOD, "answer": [], "abstained": True}
    r = _pipeline(settings, retriever, FakeLLM(payload)).ask("What about shafts in compartments?")
    assert r.confidence.label == "n/a"


def test_record_and_replay_roundtrip(tmp_path):
    from firerag.generate.providers import RecordingProvider, ReplayProvider

    path = tmp_path / "calls.jsonl"
    rec = RecordingProvider(FakeLLM(GOOD), path)
    msgs = [{"role": "user", "content": "q"}]
    first = rec.complete(msgs, 0.7, n=3)
    assert ReplayProvider(path, "fake", "scripted").complete(msgs, 0.7, n=3) == first


def test_provider_error_fails_safe(settings, retriever):
    class Down:
        name, model = "down", "x"

        def complete(self, messages, temperature, n=1):
            raise TimeoutError("provider unavailable")

    r = _pipeline(settings, retriever, Down()).ask("Do compartment shafts need protection?")
    assert r.abstained and r.answer_text == ABSTAIN_MESSAGE and r.evidence
    assert "could not be reached" in r.abstention_reason
