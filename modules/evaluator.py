"""Lightweight, dependency-free RAG evaluation.

Runs independently of the UI:

    python -m modules.evaluator                # real embeddings (+ Groq if GROQ_API_KEY is set)
    python -m modules.evaluator --offline      # smoke test with lexical hashing embeddings, no LLM
    python -m modules.evaluator --k 3 --output reports/eval.md

Metrics (all computed from an actual run - nothing is hardcoded):
- Precision@K  : share of the K retrieved chunks that contain a ground-truth evidence phrase.
- Recall@K     : share of the ground-truth evidence phrases found in at least one retrieved chunk.
- Hit rate     : share of answerable questions with at least one relevant chunk in the top K.
- Mean retrieval score : average cosine similarity of retrieved chunks (context relevance proxy).
- Answer F1    : token-overlap F1 between generated and expected answer (needs LLM).
- Answer similarity : embedding cosine between generated and expected answer (needs LLM).
- Faithfulness (lexical) : share of answer sentences whose words and numbers appear in the context.
- Faithfulness (LLM judge) : optional, an LLM rates support as SUPPORTED/PARTIAL/UNSUPPORTED.
- Abstention accuracy : for unanswerable questions, did the system say evidence was not found?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Allow `python modules/evaluator.py` as well as `python -m modules.evaluator`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import EVAL_DATASET_PATH, SAMPLE_DOCS_DIR, Settings, get_settings  # noqa: E402
from modules.ingestion import IngestReport, ingest_files  # noqa: E402
from modules.llm_client import LLMError, call_llm, get_llm  # noqa: E402
from modules.rag_pipeline import RAGPipeline  # noqa: E402
from modules.vector_store import DocumentIndex, RetrievedChunk  # noqa: E402
from utils.prompts import FAITHFULNESS_JUDGE_PROMPT  # noqa: E402

EVAL_CHUNK_SIZE = 300  # small chunks so the tiny sample corpus has enough chunks for Precision@K to mean something
EVAL_CHUNK_OVERLAP = 60

LIMITATIONS = [
    "The dataset is tiny (a few questions over two synthetic papers), so scores have large uncertainty and say little about real-world papers.",
    "Relevance ground truth is a list of evidence phrases; a relevant chunk that paraphrases the fact without the phrase is counted as irrelevant.",
    "Answer F1 and embedding similarity reward word overlap, not correctness: a correct answer worded differently scores low, and a fluent wrong answer can score high.",
    "Lexical faithfulness only checks that words and numbers occur in the context. It cannot detect wrong reasoning or negation errors.",
    "The optional LLM judge is itself a language model; its verdicts can be wrong and vary between runs.",
    "Offline mode uses hashing embeddings (word matching, not semantics); its retrieval scores must not be quoted as the system's real retrieval quality.",
]

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were which with "
    "does did how what why who when where than then also into per using used use".split()
)
_TOKEN_RE = re.compile(r"\d[\d,.]*\d|\d|[a-z]+")
_BRACKET_RE = re.compile(r"[\[\u3010][^\]\u3011]*[\]\u3011]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# --------------------------------------------------------------------- data


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_answer: str
    relevant_source: str | None
    evidence_phrases: list[str]
    answerable: bool = True


def load_dataset(path: Path | str = EVAL_DATASET_PATH) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            relevant_source=item.get("relevant_source"),
            evidence_phrases=[p.lower() for p in item.get("evidence_phrases", [])],
            answerable=bool(item.get("answerable", True)),
        )
        for item in raw["questions"]
    ]


# ------------------------------------------------------------------ metrics


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def chunk_is_relevant(chunk_text: str, evidence_phrases: list[str]) -> bool:
    normalized = _normalize(chunk_text)
    return any(phrase in normalized for phrase in evidence_phrases)


def precision_at_k(retrieved_texts: list[str], evidence_phrases: list[str]) -> float | None:
    """Fraction of retrieved chunks that are relevant (None if there is no ground truth)."""
    if not evidence_phrases:
        return None
    if not retrieved_texts:
        return 0.0
    relevant = sum(chunk_is_relevant(t, evidence_phrases) for t in retrieved_texts)
    return relevant / len(retrieved_texts)


def recall_at_k(retrieved_texts: list[str], evidence_phrases: list[str]) -> float | None:
    """Fraction of ground-truth evidence phrases covered by the retrieved chunks."""
    if not evidence_phrases:
        return None
    joined = _normalize(" ".join(retrieved_texts))
    covered = sum(phrase in joined for phrase in evidence_phrases)
    return covered / len(evidence_phrases)


def strip_citations(text: str) -> str:
    """Remove [S1] / [file.pdf, p.2] style citation brackets before scoring."""
    return re.sub(r"\s+", " ", _BRACKET_RE.sub("", text)).strip()


def _tokens(text: str) -> list[str]:
    return [t.strip(",.") for t in _TOKEN_RE.findall(text.lower())]


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words F1 between two answers (stop words removed)."""
    pred = [t for t in _tokens(prediction) if t not in _STOPWORDS]
    ref = [t for t in _tokens(reference) if t not in _STOPWORDS]
    if not pred or not ref:
        return 0.0
    common = 0
    ref_pool = list(ref)
    for token in pred:
        if token in ref_pool:
            ref_pool.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(embeddings: Embeddings, text_a: str, text_b: str) -> float:
    """Cosine similarity of two texts under the given embedding model."""
    a = np.asarray(embeddings.embed_query(text_a), dtype="float32")
    b = np.asarray(embeddings.embed_query(text_b), dtype="float32")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


def lexical_faithfulness(answer: str, context: str, support_threshold: float = 0.7) -> float | None:
    """Share of answer sentences supported by the context.

    A sentence is supported when every number in it appears in the context and
    at least ``support_threshold`` of its content words do. Returns None when
    the answer has no scorable sentences.
    """
    context_tokens = set(_tokens(context))
    sentences = [s for s in _SENTENCE_RE.split(strip_citations(answer)) if s.strip()]
    scored = supported = 0
    for sentence in sentences:
        words = [t for t in _tokens(sentence) if t not in _STOPWORDS]
        if not words:
            continue
        scored += 1
        numbers = [w for w in words if any(ch.isdigit() for ch in w)]
        if any(n not in context_tokens for n in numbers):
            continue
        coverage = sum(w in context_tokens for w in words) / len(words)
        if coverage >= support_threshold:
            supported += 1
    return supported / scored if scored else None


def llm_faithfulness(llm: BaseChatModel, answer: str, context: str) -> float | None:
    """Ask an LLM to judge support: SUPPORTED=1, PARTIAL=0.5, UNSUPPORTED=0."""
    try:
        verdict = call_llm(
            llm, FAITHFULNESS_JUDGE_PROMPT, {"context": context, "answer": strip_citations(answer)}
        )
    except LLMError:
        return None
    first_word = verdict.strip().split()[0].upper().strip(".:,") if verdict.strip() else ""
    return {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0}.get(first_word)


# ------------------------------------------------------------------- report


@dataclass
class QuestionResult:
    question_id: str
    question: str
    answerable: bool
    retrieved: list[dict] = field(default_factory=list)
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    hit: bool | None = None
    mean_retrieval_score: float | None = None
    answer: str | None = None
    answer_f1: float | None = None
    answer_similarity: float | None = None
    faithfulness_lexical: float | None = None
    faithfulness_judge: float | None = None
    abstained: bool | None = None
    error: str | None = None


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


@dataclass
class EvaluationReport:
    k: int
    settings_info: dict
    results: list[QuestionResult]

    def aggregates(self) -> dict[str, float | None]:
        answerable = [r for r in self.results if r.answerable]
        unanswerable = [r for r in self.results if not r.answerable and r.abstained is not None]
        return {
            f"Precision@{self.k}": _mean([r.precision_at_k for r in answerable]),
            f"Recall@{self.k}": _mean([r.recall_at_k for r in answerable]),
            f"Hit rate@{self.k}": _mean([None if r.hit is None else float(r.hit) for r in answerable]),
            "Mean retrieval score": _mean([r.mean_retrieval_score for r in self.results]),
            "Answer F1 (token overlap)": _mean([r.answer_f1 for r in answerable]),
            "Answer similarity (embedding)": _mean([r.answer_similarity for r in answerable]),
            "Faithfulness (lexical)": _mean([r.faithfulness_lexical for r in answerable]),
            "Faithfulness (LLM judge)": _mean([r.faithfulness_judge for r in answerable]),
            "Abstention accuracy": _mean([float(r.abstained) for r in unanswerable]),
        }

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "settings": self.settings_info,
            "aggregates": self.aggregates(),
            "results": [r.__dict__ for r in self.results],
        }

    def to_markdown(self) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines = ["# ResearchMate evaluation report", ""]
        lines += [f"- **{key}**: {value}" for key, value in self.settings_info.items()]
        lines += ["", "## Aggregate metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {name} | {fmt(value)} |" for name, value in self.aggregates().items()]
        lines += ["", "## Per-question results", ""]
        lines += [
            f"| ID | Answerable | P@{self.k} | R@{self.k} | Hit | F1 | Sim | Faith(lex) | Faith(judge) | Abstained |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.question_id} | {r.answerable} | {fmt(r.precision_at_k)} | {fmt(r.recall_at_k)} | "
                f"{r.hit if r.hit is not None else 'n/a'} | {fmt(r.answer_f1)} | {fmt(r.answer_similarity)} | "
                f"{fmt(r.faithfulness_lexical)} | {fmt(r.faithfulness_judge)} | "
                f"{r.abstained if r.abstained is not None else 'n/a'} |"
            )
        errors = [r for r in self.results if r.error]
        if errors:
            lines += ["", "## Errors", ""] + [f"- {r.question_id}: {r.error}" for r in errors]
        lines += ["", "## Limitations of this evaluation", ""] + [f"- {text}" for text in LIMITATIONS]
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- runner


def build_eval_index(
    embeddings: Embeddings,
    settings: Settings,
    docs_dir: Path | str = SAMPLE_DOCS_DIR,
    chunk_size: int = EVAL_CHUNK_SIZE,
    chunk_overlap: int = EVAL_CHUNK_OVERLAP,
) -> tuple[DocumentIndex, list[IngestReport]]:
    """Index every PDF in ``docs_dir`` into a fresh, separate index."""
    index = DocumentIndex(embeddings)
    files = [(p.name, p.read_bytes()) for p in sorted(Path(docs_dir).glob("*.pdf"))]
    if not files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}. Run scripts/generate_sample_documents.py")
    reports = ingest_files(files, index, settings, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return index, reports


def run_evaluation(
    index: DocumentIndex,
    questions: list[EvalQuestion],
    settings: Settings,
    k: int = 3,
    llm: BaseChatModel | None = None,
    use_llm_judge: bool = False,
    extra_info: dict | None = None,
    delay_seconds: float = 0.0,
) -> EvaluationReport:
    """Evaluate retrieval (always) and generation (when an LLM is supplied)."""
    pipeline = RAGPipeline(index, llm, settings)
    results: list[QuestionResult] = []

    for item in questions:
        chunks: list[RetrievedChunk] = pipeline.retrieve(item.question, k=k)
        texts = [c.text for c in chunks]
        result = QuestionResult(
            question_id=item.id,
            question=item.question,
            answerable=item.answerable,
            retrieved=[
                {
                    "source": c.source,
                    "page": c.page,
                    "score": round(c.score, 4),
                    "relevant": chunk_is_relevant(c.text, item.evidence_phrases) if item.evidence_phrases else None,
                }
                for c in chunks
            ],
            mean_retrieval_score=_mean([c.score for c in chunks]),
        )
        if item.answerable:
            result.precision_at_k = precision_at_k(texts, item.evidence_phrases)
            result.recall_at_k = recall_at_k(texts, item.evidence_phrases)
            result.hit = any(chunk_is_relevant(t, item.evidence_phrases) for t in texts)

        if llm is not None:
            generation = pipeline.generate(item.question, chunks) if chunks else None
            if generation is None:
                result.abstained = True
                result.answer = "(no chunks retrieved)"
            elif generation.error:
                result.error = generation.error
            else:
                result.answer = generation.answer
                result.abstained = generation.insufficient_evidence
                context = " ".join(texts)
                if item.answerable and not generation.insufficient_evidence:
                    cleaned = strip_citations(generation.answer)
                    result.answer_f1 = token_f1(cleaned, item.expected_answer)
                    result.answer_similarity = embedding_similarity(
                        index.embeddings, cleaned, item.expected_answer
                    )
                    result.faithfulness_lexical = lexical_faithfulness(generation.answer, context)
                    if use_llm_judge:
                        result.faithfulness_judge = llm_faithfulness(llm, generation.answer, context)
                elif item.answerable and generation.insufficient_evidence:
                    # Abstaining on an answerable question counts as a failed answer.
                    result.answer_f1 = 0.0
                    result.answer_similarity = 0.0
            if delay_seconds:
                time.sleep(delay_seconds)
        results.append(result)

    info = {
        "Questions": len(questions),
        "Chunking": f"{EVAL_CHUNK_SIZE} chars / {EVAL_CHUNK_OVERLAP} overlap (evaluation setting)",
        "Generation metrics": "computed with an LLM" if llm is not None else "skipped (no LLM)",
    }
    info.update(extra_info or {})
    return EvaluationReport(k=k, settings_info=info, results=results)


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ResearchMate RAG pipeline.")
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per question")
    parser.add_argument("--offline", action="store_true", help="use lexical hashing embeddings (no model download) and no LLM")
    parser.add_argument("--no-llm", action="store_true", help="evaluate retrieval only")
    parser.add_argument("--judge", action="store_true", help="also run the LLM faithfulness judge")
    parser.add_argument("--dataset", default=str(EVAL_DATASET_PATH))
    parser.add_argument("--output", default=None, help="write the Markdown report to this path")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.offline:
        from modules.embeddings import HashingEmbeddings

        embeddings: Embeddings = HashingEmbeddings()
        embedding_label = "HashingEmbeddings (offline smoke test; NOT semantic)"
    else:
        from modules.embeddings import get_embeddings

        embeddings = get_embeddings(settings.embedding_model)
        embedding_label = settings.embedding_model

    llm = None
    if not (args.offline or args.no_llm):
        try:
            llm = get_llm(settings)
        except LLMError as exc:
            print(f"Note: {exc} Running retrieval-only evaluation.", file=sys.stderr)

    index, reports = build_eval_index(embeddings, settings)
    for report in reports:
        print(f"[{report.status}] {report.name}: {report.message}", file=sys.stderr)

    report = run_evaluation(
        index,
        load_dataset(args.dataset),
        settings,
        k=args.k,
        llm=llm,
        use_llm_judge=args.judge,
        extra_info={
            "Embedding model": embedding_label,
            "LLM": settings.groq_model if llm is not None else "none",
        },
        delay_seconds=1.0 if llm is not None else 0.0,
    )
    markdown = report.to_markdown()
    print(markdown)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
