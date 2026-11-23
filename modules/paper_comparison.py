"""Compare two or more papers aspect by aspect using retrieved evidence."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from config.settings import Settings
from modules.rag_pipeline import AnalysisResult, run_document_analysis
from modules.vector_store import DocumentIndex
from utils.prompts import COMPARISON_PROMPT, COMPARISON_QUERIES


def compare_papers(
    index: DocumentIndex,
    llm: BaseChatModel | None,
    paper_names: list[str],
    settings: Settings,
) -> AnalysisResult:
    """Structured comparison of the selected papers (needs at least two)."""
    unique_names = list(dict.fromkeys(paper_names))
    if len(unique_names) < 2:
        return AnalysisResult("Paper comparison", "", error="Select at least two different papers to compare.")
    return run_document_analysis(
        index=index,
        llm=llm,
        sources=unique_names,
        queries=COMPARISON_QUERIES,
        prompt=COMPARISON_PROMPT,
        title="Paper comparison",
        settings=settings,
        per_query_k=2,
    )
