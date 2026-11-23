"""Embedding models.

Production: a local Sentence Transformers model (default all-MiniLM-L6-v2,
384-dimensional, ~90 MB). It runs on CPU and needs no paid API.

Offline/testing: HashingEmbeddings, a deterministic bag-of-words model that
needs no download. It is NOT semantic (it only matches shared words), so it is
used for unit tests and a smoke-test mode of the evaluator, never as the real
embedding model. Scores produced with it must be labelled accordingly.

Both return L2-normalised vectors, so inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class HashingEmbeddings(Embeddings):
    """Deterministic lexical embeddings via the hashing trick (test/offline use)."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype="float32")
        for token in _TOKEN_RE.findall(text.lower()):
            # md5 is used only as a stable hash (Python's hash() is salted per process).
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> Embeddings:
    """Load (and cache) the local Sentence Transformers embedding model.

    The first call downloads the model from the Hugging Face Hub and caches it
    on disk; later calls reuse the in-memory object.
    """
    # Imported here so unit tests that use HashingEmbeddings do not need to load torch.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine via dot product
    )
