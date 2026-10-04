"""Explainable RAG over UK fire-safety guidance (Approved Document B)."""

# On macOS, faiss and torch each bundle an OpenMP runtime. Importing faiss first and
# then running torch inference crashes the process without a traceback, so torch is
# always loaded first.
import torch  # noqa: F401

__version__ = "0.1.0"
