"""A small FAISS-backed index that keeps chunk metadata alongside the vectors.

We call FAISS directly instead of LangChain's FAISS wrapper (which lives in the
langchain-community package that is being sunset). FAISS only stores vectors
and returns integer positions, so we keep a parallel Python list of Documents:
vector i in the index <-> self._documents[i].

Index type: IndexFlatIP (exact inner-product search). With L2-normalised
vectors, inner product equals cosine similarity, so a higher score means more
similar (range about -1..1). Exact search is fast enough for the few thousand
chunks a session produces; approximate indexes (IVF, HNSW) only pay off at
far larger scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class RetrievedChunk:
    """A chunk returned by a search, together with its similarity score."""

    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self) -> int:
        return int(self.document.metadata.get("page", 0))

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def chunk_id(self) -> str:
        return self.document.metadata.get("chunk_id", "")


@dataclass
class DocumentInfo:
    """Registry entry for one ingested document."""

    name: str
    doc_hash: str
    pages: int
    chunks: int
    chars: int
    warnings: list[str] = field(default_factory=list)


class DocumentIndex:
    """In-memory vector index for all documents of one user session."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._index: faiss.Index | None = None
        self._documents: list[Document] = []
        self._registry: dict[str, DocumentInfo] = {}

    # ------------------------------------------------------------ registry
    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def chunk_count(self) -> int:
        return len(self._documents)

    @property
    def documents(self) -> list[DocumentInfo]:
        return list(self._registry.values())

    def document_names(self) -> list[str]:
        return [info.name for info in self._registry.values()]

    def has_document(self, doc_hash: str) -> bool:
        return doc_hash in self._registry

    def register_document(self, info: DocumentInfo) -> None:
        self._registry[info.doc_hash] = info

    # -------------------------------------------------------------- indexing
    def add_chunks(self, chunks: list[Document]) -> int:
        """Embed chunks and add them to the index. Returns the number added."""
        if not chunks:
            return 0
        vectors = np.asarray(
            self._embeddings.embed_documents([c.page_content for c in chunks]), dtype="float32"
        )
        if vectors.ndim != 2:
            raise ValueError("Embedding model returned an unexpected shape")
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != self._index.d:
            raise ValueError("Embedding dimension changed; clear the session and re-upload")
        self._index.add(vectors)
        self._documents.extend(chunks)
        return len(chunks)

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        k: int = 5,
        sources: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks to ``query``.

        ``sources`` restricts the search to the named documents. FAISS cannot
        filter by itself, so when a filter is given we ask for *all* vectors
        (exact search, cheap at this scale) and filter the results in Python.
        ``min_score`` drops chunks below a cosine-similarity threshold.
        """
        if self._index is None or not self._documents or k < 1:
            return []
        query_vector = np.asarray([self._embeddings.embed_query(query)], dtype="float32")
        fetch = self._index.ntotal if sources else min(k, self._index.ntotal)
        scores, positions = self._index.search(query_vector, fetch)

        allowed = set(sources) if sources else None
        results: list[RetrievedChunk] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when fewer results exist
                continue
            document = self._documents[int(position)]
            if allowed is not None and document.metadata.get("source") not in allowed:
                continue
            if float(score) < min_score:
                continue
            results.append(RetrievedChunk(document=document, score=float(score)))
            if len(results) >= k:
                break
        return results

    def chunks_for_source(self, source: str) -> list[Document]:
        """All chunks of one document (used by tests and the evaluator)."""
        return [d for d in self._documents if d.metadata.get("source") == source]
