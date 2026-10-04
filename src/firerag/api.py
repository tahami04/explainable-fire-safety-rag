"""FastAPI service.

Run:
    uvicorn firerag.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from firerag import __version__
from firerag.config import get_settings
from firerag.governance.audit import write_feedback
from firerag.pipeline import AskResult, Pipeline

app = FastAPI(
    title="Explainable Fire-Safety RAG",
    version=__version__,
    description="Source-attributed answers over Approved Document B with faithfulness checks, "
    "confidence scoring and abstention. Demonstration only; not professional advice.",
)


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=3, max_length=1000)
    taxonomy_filter: list[str] = Field(default_factory=list, description="e.g. ['B3', 'B1.escape_routes']")
    mode: Literal["full", "retrieval_only"] | None = None


class FeedbackRequest(BaseModel):
    """Request body for ``POST /feedback``."""

    query_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """The shared pipeline (models and index loaded once)."""
    return Pipeline()


@app.get("/health")
def health() -> dict:
    """Liveness and configuration summary (no secrets)."""
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_available": s.llm_available,
        "provider": s.llm_provider if s.llm_available else None,
    }


@app.post("/ask", response_model=AskResult)
def ask(req: AskRequest, pipeline: Pipeline = Depends(get_pipeline)) -> AskResult:
    """Answer a question with citations, evidence, confidence and reasoning pathway."""
    try:
        return pipeline.ask(req.question, taxonomy_filter=req.taxonomy_filter, mode=req.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Record pilot feedback for a previous answer."""
    return write_feedback(get_settings().log_dir, req.query_id, req.rating, req.comment)
