from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from firerag import api
from firerag.config import get_settings
from firerag.pipeline import Pipeline

from .conftest import FakeLLM, entail_all, fake_embed
from .test_abstention import GOOD


@pytest.fixture
def client(settings, retriever, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(settings.log_dir))
    get_settings.cache_clear()
    pipe = Pipeline(settings, retriever=retriever, provider=FakeLLM(GOOD), nli_scorer=entail_all, embed=fake_embed)
    api.app.dependency_overrides[api.get_pipeline] = lambda: pipe
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "llm_available" in body


def test_ask_returns_full_schema(client):
    resp = client.post("/ask", json={"question": "Do compartment shafts need protection?"})
    assert resp.status_code == 200
    body = resp.json()
    for key in [
        "query_id",
        "answer",
        "answer_text",
        "reasoning_pathway",
        "caveats",
        "abstained",
        "confidence",
        "consistency",
        "evidence",
        "model",
        "prompt_version",
        "latency_ms",
    ]:
        assert key in body
    sent = body["answer"][0]
    assert {"sentence", "citations", "citation_labels", "faithfulness", "quotes"} <= set(sent)
    ev = body["evidence"][0]
    assert {"chunk_id", "citation", "page", "text", "explanation"} <= set(ev)
    assert {"bm25_score", "dense_similarity", "fused_rank", "matched_terms"} <= set(ev["explanation"])
    assert set(body["confidence"]) >= {"retrieval_strength", "faithfulness_rate", "consistency", "overall", "label"}


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "ok?", "mode": "bogus"}).status_code == 422


def test_feedback_is_logged_and_redacted(client, settings):
    resp = client.post(
        "/feedback", json={"query_id": "abc", "rating": "down", "comment": "wrong, email me at a.b@example.com"}
    )
    assert resp.status_code == 200
    line = json.loads((settings.log_dir / "feedback.jsonl").read_text().splitlines()[-1])
    assert line["rating"] == "down" and "[email]" in line["comment"] and "example.com" not in line["comment"]
