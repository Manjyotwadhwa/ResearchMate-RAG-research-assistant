"""Tests for evaluation metrics and the runnable evaluator (offline mode)."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config.settings import Settings
from modules.embeddings import HashingEmbeddings
from modules.evaluator import (
    build_eval_index,
    lexical_faithfulness,
    llm_faithfulness,
    load_dataset,
    main,
    precision_at_k,
    recall_at_k,
    run_evaluation,
    strip_citations,
    token_f1,
)


def test_precision_and_recall_known_values():
    texts = ["the model has 1.2 million parameters", "unrelated text", "also unrelated"]
    phrases = ["1.2 million parameters", "adam optimizer"]
    assert precision_at_k(texts, phrases) == pytest.approx(1 / 3)
    assert recall_at_k(texts, phrases) == pytest.approx(1 / 2)


def test_metrics_without_ground_truth_are_none():
    assert precision_at_k(["x"], []) is None
    assert recall_at_k(["x"], []) is None


def test_precision_with_no_retrieval_is_zero():
    assert precision_at_k([], ["x"]) == 0.0


def test_token_f1():
    assert token_f1("LeafLite has 1.2 million parameters", "LeafLite has 1.2 million parameters.") == pytest.approx(1.0)
    assert token_f1("completely different words", "LeafLite has 1.2 million parameters") == 0.0
    assert 0 < token_f1("LeafLite has 1.2 million", "LeafLite has 1.2 million parameters") < 1


def test_lexical_faithfulness_detects_unsupported_numbers():
    context = "LeafLite reaches 94.1 percent accuracy on the VillageLeaf dataset."
    assert lexical_faithfulness("LeafLite reaches 94.1 percent accuracy.", context) == 1.0
    assert lexical_faithfulness("LeafLite reaches 99.9 percent accuracy.", context) == 0.0


def test_lexical_faithfulness_mixed_sentences():
    context = "The model uses the Adam optimizer."
    answer = "The model uses the Adam optimizer. It was published in Nature by famous authors."
    assert lexical_faithfulness(answer, context) == pytest.approx(0.5)


def test_strip_citations():
    assert strip_citations("Claim [a.pdf, p.1] more [S2].") == "Claim more ."


def test_llm_judge_parsing():
    assert llm_faithfulness(FakeListChatModel(responses=["SUPPORTED"]), "a", "b") == 1.0
    assert llm_faithfulness(FakeListChatModel(responses=["Partial."]), "a", "b") == 0.5
    assert llm_faithfulness(FakeListChatModel(responses=["???"]), "a", "b") is None


def test_dataset_loads_and_is_well_formed():
    questions = load_dataset()
    assert len(questions) >= 8
    assert any(not q.answerable for q in questions)
    assert all(q.evidence_phrases for q in questions if q.answerable)


@pytest.fixture(scope="module")
def eval_setup():
    settings = Settings(groq_api_key="x")
    index, reports = build_eval_index(HashingEmbeddings(), settings)
    assert all(r.status == "added" for r in reports)
    return settings, index


def test_retrieval_only_evaluation_runs_on_sample_documents(eval_setup):
    settings, index = eval_setup
    report = run_evaluation(index, load_dataset(), settings, k=3)
    aggregates = report.aggregates()
    assert 0.0 <= aggregates["Precision@3"] <= 1.0
    assert 0.0 <= aggregates["Recall@3"] <= 1.0
    assert aggregates["Answer F1 (token overlap)"] is None  # no LLM -> not computed, not invented
    assert "Aggregate metrics" in report.to_markdown()


def test_evaluation_with_fake_llm_scores_abstention(eval_setup):
    settings, index = eval_setup
    llm = FakeListChatModel(responses=["Sufficient evidence was not found in the uploaded documents."])
    report = run_evaluation(index, load_dataset(), settings, k=3, llm=llm)
    unanswerable = [r for r in report.results if not r.answerable][0]
    assert unanswerable.abstained is True
    assert report.aggregates()["Abstention accuracy"] == 1.0
    # abstaining on answerable questions is scored as a failed answer
    assert report.aggregates()["Answer F1 (token overlap)"] == 0.0


def test_cli_offline_runs_and_writes_report(tmp_path, capsys):
    output = tmp_path / "report.md"
    assert main(["--offline", "--k", "3", "--output", str(output)]) == 0
    assert "NOT semantic" in output.read_text(encoding="utf-8")
