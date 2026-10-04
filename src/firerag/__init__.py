"""Explainable RAG over UK fire-safety guidance (Approved Document B)."""

import os
import sys

if sys.platform == "darwin":
    # The macOS pip wheels of torch and faiss-cpu each bundle their own libomp, and
    # OpenMP aborts when a second copy initialises. We allow the duplicate, keep
    # faiss's OpenMP single-threaded, and verify faiss results against an exact NumPy
    # computation every time an index is loaded (see retrieve/index.py). Linux
    # (Docker, CI) is unaffected.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(os.cpu_count() or 1)

__version__ = "0.1.0"
