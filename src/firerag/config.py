"""Central configuration, read from environment variables / ``.env``.

All thresholds live here so they are visible, documented and easy to change.
See docs/ARCHITECTURE.md ("How the thresholds were chosen").
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings. Every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- paths -----------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / "data" / "index"
    log_dir: Path = ROOT / "logs"
    taxonomy_path: Path = ROOT / "taxonomy" / "fire_safety_taxonomy.yaml"

    # --- models (all local, no API key needed) --------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    use_reranker: bool = True

    # --- LLM provider -----------------------------------------------------------
    llm_provider: Literal["openai", "ollama", "none"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"
    llm_timeout_s: float = 60.0
    prompt_version: str = "grounded-v2"

    # --- retrieval ----------------------------------------------------------------
    candidates_per_retriever: int = 30
    rrf_k: int = 60
    rerank_pool: int = 20
    top_k: int = 6

    # --- reliability layer --------------------------------------------------------
    consistency_samples: int = 5
    sample_temperature: float = 0.7
    # Pre-generation gate: abstain without calling the LLM if the best reranked
    # passage has relevance probability below this.
    min_retrieval_strength: float = 0.10
    # NLI entailment probability bands for per-sentence faithfulness labels.
    nli_supported: float = 0.50
    nli_partial: float = 0.15
    # Weights for the overall confidence score (must sum to 1).
    w_retrieval: float = 0.30
    w_faithfulness: float = 0.40
    w_consistency: float = 0.30
    conf_high: float = 0.75
    conf_medium: float = 0.50
    # Post-generation abstention: overall score below this ⇒ abstain.
    abstain_below: float = 0.35

    @property
    def llm_available(self) -> bool:
        """True when a generation provider is configured and usable."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return self.llm_provider == "ollama"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


ABSTAIN_MESSAGE = "Insufficient evidence in the indexed guidance; refer to a competent fire engineer."
