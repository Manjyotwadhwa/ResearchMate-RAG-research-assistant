"""Structured summary of a single paper (fields missing from the paper stay 'not found')."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from config.settings import Settings
from modules.rag_pipeline import AnalysisResult, run_document_analysis
from modules.vector_store import DocumentIndex
from utils.prompts import SUMMARY_PROMPT, SUMMARY_QUERIES


def summarize_paper(
    index: DocumentIndex,
    llm: BaseChatModel | None,
    paper_name: str,
    settings: Settings,
) -> AnalysisResult:
    """Summarise one paper from evidence retrieved for each summary field."""
    if paper_name not in index.document_names():
        return AnalysisResult("Paper summary", "", error=f"'{paper_name}' is not in the current session.")
    return run_document_analysis(
        index=index,
        llm=llm,
        sources=[paper_name],
        queries=SUMMARY_QUERIES,
        prompt=SUMMARY_PROMPT,
        title=f"Summary: {paper_name}",
        settings=settings,
        per_query_k=2,
    )
