"""Retrieval-Augmented Generation: retrieve -> build prompt -> generate -> cite.

Retrieval and generation are separate methods on purpose (``retrieve`` and
``generate``) so each can be tested and evaluated alone. The same module also
holds the evidence-gathering helpers that comparison, summarisation and gap
analysis share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from config.settings import Settings
from modules.llm_client import LLMError, call_llm
from modules.vector_store import DocumentIndex, RetrievedChunk
from utils.helpers import (
    extract_cited_ids,
    format_citation,
    resolve_citations,
    sanitize_question,
)
from utils.prompts import NO_EVIDENCE_MESSAGE, RAG_PROMPT

# --------------------------------------------------------------------- Q&A


@dataclass
class RAGResult:
    question: str
    answer: str  # citations already resolved to "file.pdf, p.N"
    chunks: list[RetrievedChunk]
    cited_labels: list[str] = field(default_factory=list)  # passages the model actually cited
    unverified_citations: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    error: str | None = None


def format_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, str]]:
    """Number the retrieved chunks as S1..Sn and build the prompt context.

    Returns (context_text, labels) where labels maps 'S1' -> 'paper.pdf, p.3'.
    The model only ever sees the tag; we map it back to the real document and
    page ourselves, so a citation can never point at something that was not retrieved.
    """
    labels: dict[str, str] = {}
    blocks: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        tag = f"S{number}"
        labels[tag] = format_citation(chunk.source, chunk.page)
        blocks.append(f"[{tag}] (document: {chunk.source}, page {chunk.page})\n{chunk.text}")
    return "\n\n".join(blocks), labels


class RAGPipeline:
    """Question answering over the indexed documents."""

    def __init__(
        self,
        index: DocumentIndex,
        llm: BaseChatModel | None,
        settings: Settings,
    ) -> None:
        self.index = index
        self.llm = llm
        self.settings = settings

    # ---- step 1: retrieval (no LLM involved)
    def retrieve(
        self, question: str, k: int | None = None, sources: list[str] | None = None
    ) -> list[RetrievedChunk]:
        return self.index.search(
            question,
            k=k or self.settings.top_k,
            sources=sources,
            min_score=self.settings.min_relevance_score,
        )

    # ---- step 2: generation (LLM only, given the retrieved chunks)
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> RAGResult:
        context, labels = format_context(chunks)
        if self.llm is None:
            return RAGResult(
                question, "", chunks, error="No language model configured (GROQ_API_KEY is missing)."
            )
        try:
            raw_answer = call_llm(self.llm, RAG_PROMPT, {"context": context, "question": question})
        except LLMError as exc:
            return RAGResult(question, "", chunks, error=str(exc))

        cited_ids = extract_cited_ids(raw_answer)
        answer, unverified = resolve_citations(raw_answer, labels)
        return RAGResult(
            question=question,
            answer=answer,
            chunks=chunks,
            cited_labels=[labels[c] for c in cited_ids if c in labels],
            unverified_citations=unverified,
            insufficient_evidence=NO_EVIDENCE_MESSAGE.lower() in raw_answer.lower(),
        )

    # ---- both steps together
    def answer(
        self, question: str, k: int | None = None, sources: list[str] | None = None
    ) -> RAGResult:
        try:
            question = sanitize_question(question, self.settings.max_question_chars)
        except ValueError as exc:
            return RAGResult(question or "", "", [], error=str(exc))

        chunks = self.retrieve(question, k=k, sources=sources)
        if not chunks:
            # Nothing relevant was retrieved: do not ask the LLM to guess.
            return RAGResult(question, NO_EVIDENCE_MESSAGE, [], insufficient_evidence=True)
        return self.generate(question, chunks)


# ------------------------------------- evidence gathering for multi-paper tasks


@dataclass
class EvidenceItem:
    evidence_id: str  # 'E1', 'E2', ...
    chunk: RetrievedChunk

    @property
    def label(self) -> str:
        return format_citation(self.chunk.source, self.chunk.page)


@dataclass
class AnalysisResult:
    """Output of comparison, summarisation and gap analysis."""

    title: str
    text: str  # citations resolved to "file.pdf, p.N"
    evidence: list[EvidenceItem] = field(default_factory=list)
    unverified_citations: list[str] = field(default_factory=list)
    error: str | None = None


def gather_evidence(
    index: DocumentIndex,
    source: str,
    queries: list[str],
    per_query_k: int,
    max_chars: int,
) -> list[RetrievedChunk]:
    """Collect diverse chunks from ONE document, one retrieval query per aspect.

    Chunks are taken round-robin (best hit of every query first, then second-best,
    ...) so every aspect is represented even when the character budget is small.
    Duplicates are skipped. The result is sorted into reading order (by page).
    """
    per_query = [index.search(q, k=per_query_k, sources=[source]) for q in queries]
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()
    used_chars = 0
    for rank in range(per_query_k):
        for hits in per_query:
            if rank >= len(hits):
                continue
            chunk = hits[rank]
            if chunk.chunk_id in seen:
                continue
            if used_chars + len(chunk.text) > max_chars and selected:
                continue
            seen.add(chunk.chunk_id)
            selected.append(chunk)
            used_chars += len(chunk.text)
    selected.sort(key=lambda c: (c.page, c.document.metadata.get("chunk_index", 0)))
    return selected


def run_document_analysis(
    index: DocumentIndex,
    llm: BaseChatModel | None,
    sources: list[str],
    queries: list[str],
    prompt: ChatPromptTemplate,
    title: str,
    settings: Settings,
    per_query_k: int = 2,
    extra_variables: dict | None = None,
) -> AnalysisResult:
    """Shared engine: gather evidence per paper -> number it -> call LLM -> verify citations."""
    if llm is None:
        return AnalysisResult(title, "", error="No language model configured (GROQ_API_KEY is missing).")

    budget_per_paper = max(1500, settings.analysis_max_context_chars // max(1, len(sources)))
    evidence: list[EvidenceItem] = []
    blocks: list[str] = []
    labels: dict[str, str] = {}
    counter = 1
    for source in sources:
        chunks = gather_evidence(index, source, queries, per_query_k, budget_per_paper)
        lines = [f"=== PAPER: {source} ==="]
        for chunk in chunks:
            item = EvidenceItem(evidence_id=f"E{counter}", chunk=chunk)
            counter += 1
            evidence.append(item)
            labels[item.evidence_id] = item.label
            lines.append(f"[{item.evidence_id}] (page {chunk.page})\n{chunk.text}")
        blocks.append("\n\n".join(lines))

    if not evidence:
        return AnalysisResult(title, "", error="No text could be retrieved for the selected documents.")

    variables = {
        "evidence": "\n\n".join(blocks),
        "paper_names": ", ".join(sources),
        "paper_name": sources[0],
    }
    variables.update(extra_variables or {})
    try:
        raw = call_llm(llm, prompt, variables)
    except LLMError as exc:
        return AnalysisResult(title, "", evidence=evidence, error=str(exc))

    text, unverified = resolve_citations(raw, labels)
    return AnalysisResult(title, text, evidence=evidence, unverified_citations=unverified)
