"""Tests for the vector index, ingestion and the RAG pipeline (offline, no network).

Retrieval tests use HashingEmbeddings (lexical), so they verify the index
mechanics (metadata, filtering, top-k, deduplication), not semantic quality.
The LLM is replaced by LangChain's fake chat models.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from modules.ingestion import ingest_files
from modules.paper_comparison import compare_papers
from modules.rag_pipeline import RAGPipeline, format_context
from modules.research_gap import analyze_research_gaps
from modules.summarizer import summarize_paper
from modules.vector_store import DocumentIndex
from modules.embeddings import HashingEmbeddings
from tests.conftest import LONG_TEXT_A, make_image_only_pdf, make_pdf
from utils.helpers import extract_cited_ids, resolve_citations, sanitize_question
from utils.prompts import NO_EVIDENCE_MESSAGE


# ------------------------------------------------------------------- index
def test_search_returns_relevant_document_with_metadata(two_paper_index):
    hits = two_paper_index.search("Adam optimizer depthwise convolutions", k=2)
    assert hits[0].source == "paper_a.pdf"
    assert hits[0].page == 1
    assert hits[0].document.metadata["chunk_id"]
    assert hits[0].score > hits[-1].score or len(hits) == 1


def test_top_k_is_respected(two_paper_index):
    assert len(two_paper_index.search("model", k=1)) == 1
    assert len(two_paper_index.search("model", k=100)) == two_paper_index.chunk_count


def test_source_filter_restricts_results(two_paper_index):
    hits = two_paper_index.search("vision transformer pretrained", k=5, sources=["paper_a.pdf"])
    assert hits and all(h.source == "paper_a.pdf" for h in hits)


def test_min_score_filters_weak_matches(two_paper_index):
    assert two_paper_index.search("zzzz qqqq", k=5, min_score=0.5) == []


def test_empty_index_returns_nothing():
    assert DocumentIndex(HashingEmbeddings()).search("anything") == []


# --------------------------------------------------------------- ingestion
def test_duplicate_upload_is_not_processed_twice(settings):
    index = DocumentIndex(HashingEmbeddings())
    data = make_pdf([LONG_TEXT_A])
    first = ingest_files([("a.pdf", data)], index, settings)
    count_after_first = index.chunk_count
    second = ingest_files([("renamed.pdf", data)], index, settings)
    assert first[0].status == "added"
    assert second[0].status == "duplicate"
    assert index.chunk_count == count_after_first


def test_same_name_different_content_gets_unique_name(settings):
    index = DocumentIndex(HashingEmbeddings())
    ingest_files([("p.pdf", make_pdf([LONG_TEXT_A]))], index, settings)
    reports = ingest_files([("p.pdf", make_pdf([LONG_TEXT_A + " Extra sentence for a different hash."]))], index, settings)
    assert reports[0].name == "p (2).pdf"
    assert sorted(index.document_names()) == ["p (2).pdf", "p.pdf"]


def test_bad_files_are_rejected_without_crashing(settings):
    index = DocumentIndex(HashingEmbeddings())
    reports = ingest_files(
        [("good.pdf", make_pdf([LONG_TEXT_A])), ("scan.pdf", make_image_only_pdf()), ("x.pdf", b"nope")],
        index,
        settings,
    )
    assert [r.status for r in reports] == ["added", "rejected", "rejected"]
    assert "OCR" in reports[1].message
    assert index.document_names() == ["good.pdf"]


def test_file_count_limit(settings):
    index = DocumentIndex(HashingEmbeddings())
    files = [(f"d{i}.pdf", make_pdf([LONG_TEXT_A + f" unique {i}"])) for i in range(settings.max_files + 1)]
    reports = ingest_files(files, index, settings)
    assert reports[-1].status == "rejected"
    assert len(index.documents) == settings.max_files


# ----------------------------------------------------------- RAG pipeline
def test_rag_answer_resolves_citations_to_document_and_page(two_paper_index, settings):
    llm = FakeListChatModel(responses=["LeafLite uses the Adam optimizer [S1]."])
    result = RAGPipeline(two_paper_index, llm, settings).answer("Which optimizer does LeafLite use?")
    assert "[paper_a.pdf, p.1]" in result.answer
    assert result.cited_labels == ["paper_a.pdf, p.1"]
    assert result.unverified_citations == []
    assert not result.insufficient_evidence
    assert result.chunks


def test_fabricated_citation_is_flagged(two_paper_index, settings):
    llm = FakeListChatModel(responses=["Some claim [S99]."])
    result = RAGPipeline(two_paper_index, llm, settings).answer("Which optimizer does LeafLite use?")
    assert result.unverified_citations == ["S99"]
    assert "unverified:S99" in result.answer


def test_model_abstention_sets_insufficient_flag(two_paper_index, settings):
    llm = FakeListChatModel(responses=[NO_EVIDENCE_MESSAGE + " The papers do not mention it."])
    result = RAGPipeline(two_paper_index, llm, settings).answer("What is the stock price?")
    assert result.insufficient_evidence


def test_no_retrieval_means_no_llm_call(settings):
    class ExplodingLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise AssertionError("LLM must not be called when nothing was retrieved")

    pipeline = RAGPipeline(DocumentIndex(HashingEmbeddings()), ExplodingLLM(responses=["x"]), settings)
    result = pipeline.answer("anything")
    assert result.insufficient_evidence and result.answer == NO_EVIDENCE_MESSAGE


def test_llm_failure_is_reported_gracefully(two_paper_index, settings):
    class RateLimitError(Exception):
        pass

    class FailingLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RateLimitError("429 rate limit")

    result = RAGPipeline(two_paper_index, FailingLLM(responses=["x"]), settings).answer("Adam optimizer?")
    assert result.error and "rate limit" in result.error.lower()
    assert result.chunks  # retrieved passages are still available to show


def test_missing_llm_still_retrieves(two_paper_index, settings):
    result = RAGPipeline(two_paper_index, None, settings).answer("Adam optimizer?")
    assert result.error and "GROQ_API_KEY" in result.error
    assert result.chunks


def test_invalid_question_is_rejected(two_paper_index, settings):
    result = RAGPipeline(two_paper_index, None, settings).answer("   ")
    assert result.error


def test_format_context_numbers_chunks(two_paper_index):
    chunks = two_paper_index.search("model", k=2)
    context, labels = format_context(chunks)
    assert set(labels) == {"S1", "S2"}
    assert "[S1] (document:" in context


# ---------------------------------------------------- multi-paper analyses
def test_comparison_requires_two_papers(two_paper_index, settings):
    result = compare_papers(two_paper_index, FakeListChatModel(responses=["x"]), ["paper_a.pdf"], settings)
    assert result.error


def test_comparison_collects_evidence_from_every_paper(two_paper_index, settings):
    llm = FakeListChatModel(responses=["### Datasets\n- paper_a - Stated: VillageLeaf [E1]\n- paper_b - Stated: FieldCrop [E2]"])
    result = compare_papers(two_paper_index, llm, ["paper_a.pdf", "paper_b.pdf"], settings)
    assert not result.error
    assert {e.chunk.source for e in result.evidence} == {"paper_a.pdf", "paper_b.pdf"}
    assert "[paper_a.pdf, p.1]" in result.text
    assert result.unverified_citations == []


def test_summary_and_gap_analysis_run(two_paper_index, settings):
    llm = FakeListChatModel(responses=["### Abstract overview\nLightweight CNN [E1]"])
    summary = summarize_paper(two_paper_index, llm, "paper_a.pdf", settings)
    assert not summary.error and "paper_a.pdf" in summary.text
    gaps = analyze_research_gaps(two_paper_index, llm, ["paper_a.pdf", "paper_b.pdf"], settings)
    assert not gaps.error
    assert summarize_paper(two_paper_index, llm, "missing.pdf", settings).error


# ------------------------------------------------------------------ helpers
def test_resolve_citations_handles_groups():
    text, unknown = resolve_citations("Claim [S1, S2] and [E3].", {"S1": "a.pdf, p.1", "S2": "b.pdf, p.2"})
    assert text == "Claim [a.pdf, p.1; b.pdf, p.2] and [unverified:E3]."
    assert unknown == ["E3"]


def test_extract_cited_ids_unique_and_ordered():
    assert extract_cited_ids("x [S2] y [S1][S2] z [E4; E5]") == ["S2", "S1", "E4", "E5"]


def test_sanitize_question():
    assert sanitize_question("  hello \x00 world \n ") == "hello world"
    with pytest.raises(ValueError):
        sanitize_question("")
    with pytest.raises(ValueError):
        sanitize_question("x" * 20, max_chars=10)
