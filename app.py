"""ResearchMate - Streamlit front end.

This file only wires the UI to the modules in ``modules/``. All logic
(extraction, chunking, retrieval, prompting, evaluation) lives there.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit command.
st.set_page_config(page_title="ResearchMate", page_icon="\U0001f4da", layout="wide")


def _load_streamlit_secrets_into_env() -> None:
    """On Streamlit Cloud, secrets are exposed via st.secrets. Copy simple
    string/number values into os.environ so config.settings reads one source."""
    try:
        for key in st.secrets:
            value = st.secrets[key]
            if isinstance(value, (str, int, float)) and key not in os.environ:
                os.environ[key] = str(value)
    except Exception:
        pass  # no secrets file locally: .env is used instead


_load_streamlit_secrets_into_env()

from config.settings import EVAL_DATASET_PATH, SAMPLE_DOCS_DIR, Settings, get_settings  # noqa: E402
from modules.embeddings import get_embeddings  # noqa: E402
from modules.evaluator import (  # noqa: E402
    EVAL_CHUNK_OVERLAP,
    EVAL_CHUNK_SIZE,
    LIMITATIONS,
    build_eval_index,
    load_dataset,
    run_evaluation,
)
from modules.ingestion import ingest_files  # noqa: E402
from modules.llm_client import LLMError, get_llm  # noqa: E402
from modules.paper_comparison import compare_papers  # noqa: E402
from modules.rag_pipeline import AnalysisResult, RAGPipeline, RAGResult  # noqa: E402
from modules.research_gap import GAP_DISCLAIMER, analyze_research_gaps  # noqa: E402
from modules.summarizer import summarize_paper  # noqa: E402
from modules.vector_store import DocumentIndex  # noqa: E402
from utils.helpers import truncate  # noqa: E402

st.markdown(
    """
    <style>
    .rm-logo {font-size: 1.6rem; font-weight: 700; letter-spacing: .5px;}
    .rm-sub {color: #6b7280; margin-top: -0.6rem; font-size: .9rem;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .6rem .9rem; border-radius: .6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ settings
try:
    SETTINGS: Settings = get_settings()
except ValueError as config_error:
    st.error(f"Configuration error: {config_error}")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model (the first run downloads it)...")
def load_embeddings(model_name: str):
    return get_embeddings(model_name)


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str, api_key: str):
    """Cached LLM client; returns None when no key is configured."""
    try:
        return get_llm(SETTINGS)
    except LLMError:
        return None


LLM = load_llm(SETTINGS.groq_model, SETTINGS.groq_api_key)


# ------------------------------------------------------------- session state
def init_session() -> None:
    if "index" not in st.session_state:
        st.session_state.index = DocumentIndex(load_embeddings(SETTINGS.embedding_model))
        st.session_state.reports = []
        st.session_state.processed_upload_ids = set()
        st.session_state.history = []
        st.session_state.analysis = {}
        st.session_state.eval_report = None
        st.session_state.uploader_key = 0


def clear_session() -> None:
    uploader_key = st.session_state.get("uploader_key", 0) + 1
    for key in ("index", "reports", "processed_upload_ids", "history", "analysis", "eval_report"):
        st.session_state.pop(key, None)
    init_session()
    st.session_state.uploader_key = uploader_key  # new key resets the file_uploader widget


init_session()
INDEX: DocumentIndex = st.session_state.index

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="rm-logo">\U0001f4da ResearchMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="rm-sub">Multi-document research assistant</div>', unsafe_allow_html=True)
    st.warning(
        "Do not upload confidential or unpublished research. Text is sent to the Groq API "
        "to generate answers. Documents are kept in memory only and vanish when the session ends.",
        icon="⚠️",
    )

    with st.expander("Chunking settings", expanded=False):
        chunk_size = st.number_input("Chunk size (characters)", 200, 4000, SETTINGS.chunk_size, step=100)
        chunk_overlap = st.number_input("Chunk overlap (characters)", 0, 1000, SETTINGS.chunk_overlap, step=50)
        st.caption("Applies to documents processed after the change. Clear the session to re-process.")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 15, SETTINGS.top_k)

    uploads = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Up to {SETTINGS.max_files} files, {SETTINGS.max_file_size_mb} MB each. Scanned PDFs (images) need OCR and are not supported.",
    )

    # Process only files we have not seen on a previous rerun.
    new_files = [f for f in uploads or [] if f.file_id not in st.session_state.processed_upload_ids]
    if new_files:
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            with st.status(f"Processing {len(new_files)} file(s)...", expanded=True) as status:
                payload = [(f.name, f.getvalue()) for f in new_files]
                try:
                    new_reports = ingest_files(payload, INDEX, SETTINGS, int(chunk_size), int(chunk_overlap))
                except Exception as exc:
                    new_reports = []
                    st.error(f"Processing failed: {exc}")
                for f in new_files:
                    st.session_state.processed_upload_ids.add(f.file_id)
                st.session_state.reports.extend(new_reports)
                status.update(label="Processing finished", state="complete", expanded=False)

    st.subheader("Documents")
    if not INDEX.documents:
        st.caption("No documents yet.")
    for info in INDEX.documents:
        st.markdown(f"**{info.name}**  \n{info.pages} pages · {info.chunks} chunks")
    for report in st.session_state.reports:
        if report.status == "rejected":
            st.error(report.message)
        elif report.status == "duplicate":
            st.info(f"{report.name}: {report.message}")
        elif report.message != "Indexed successfully.":
            st.warning(f"{report.name}: {report.message}")

    if not SETTINGS.has_llm_key:
        st.error("GROQ_API_KEY is not set. Retrieval works, but answers cannot be generated.")
    if st.button("Clear session", width="stretch"):
        clear_session()
        st.rerun()

# ---------------------------------------------------------------- main header
st.title("ResearchMate")
st.write(
    "Upload research papers, ask questions grounded in their text, compare findings, "
    "and generate structured summaries. Answers cite the document and page they came from."
)
col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(INDEX.documents))
col2.metric("Pages", sum(d.pages for d in INDEX.documents))
col3.metric("Searchable chunks", INDEX.chunk_count)

tab_chat, tab_compare, tab_summary, tab_gaps, tab_eval = st.tabs(
    ["Chat with Documents", "Paper Comparison", "Paper Summarisation", "Research Gap Analysis", "Evaluation Dashboard"]
)


# ------------------------------------------------------------------ renderers
def render_rag_result(result: RAGResult) -> None:
    if result.error:
        st.error(result.error)
    else:
        if result.insufficient_evidence:
            st.warning("The documents do not appear to contain enough evidence for this question.")
        st.markdown(result.answer)
        if result.unverified_citations:
            st.warning(
                "The model cited passages that were not retrieved ("
                + ", ".join(result.unverified_citations)
                + "). Treat those citations as unreliable."
            )
    if result.chunks:
        cited = set(result.cited_labels)
        st.markdown("**Sources**")
        seen = []
        for chunk in result.chunks:
            label = f"{chunk.source}, p.{chunk.page}"
            if label not in seen:
                seen.append(label)
        for label in seen:
            st.markdown(f"- {label}" + (" ✓ cited in answer" if label in cited else ""))
        with st.expander(f"Retrieved passages ({len(result.chunks)})"):
            for number, chunk in enumerate(result.chunks, start=1):
                st.markdown(f"**S{number} · {chunk.source} · page {chunk.page} · similarity {chunk.score:.3f}**")
                st.write(chunk.text)
                st.divider()


def render_analysis(result: AnalysisResult) -> None:
    if result.error:
        st.error(result.error)
        return
    st.markdown(result.text)
    if result.unverified_citations:
        st.warning(
            "Some citations point to evidence that was not provided ("
            + ", ".join(result.unverified_citations)
            + "). Treat them as unreliable."
        )
    with st.expander(f"Evidence used ({len(result.evidence)} passages)"):
        for item in result.evidence:
            st.markdown(f"**{item.evidence_id} · {item.label}**")
            st.write(truncate(item.chunk.text, 700))


def require_documents(minimum: int = 1) -> bool:
    if len(INDEX.documents) < minimum:
        st.info(f"Upload at least {minimum} PDF{'s' if minimum > 1 else ''} in the sidebar to use this tab.")
        return False
    return True


def llm_ready() -> bool:
    if LLM is None:
        st.error("This feature needs a Groq API key. Set GROQ_API_KEY and restart the app.")
        return False
    return True


# ------------------------------------------------------------------ chat tab
with tab_chat:
    st.subheader("Ask a question across your documents")
    if require_documents(1):
        names = INDEX.document_names()
        scope = st.multiselect("Search within", names, default=names, key="chat_scope")
        with st.form("ask_form"):
            question = st.text_area(
                "Research question",
                placeholder="e.g. Which dataset was used and what accuracy was reported?",
                max_chars=SETTINGS.max_question_chars,
            )
            submitted = st.form_submit_button("Ask", type="primary")
        if submitted:
            if not scope:
                st.warning("Select at least one document to search.")
            else:
                pipeline = RAGPipeline(INDEX, LLM, SETTINGS)
                with st.spinner("Retrieving passages and generating an answer..."):
                    result = pipeline.answer(question, k=top_k, sources=scope)
                st.session_state.history.insert(0, result)
        if st.session_state.history:
            st.markdown(f"#### {st.session_state.history[0].question}")
            render_rag_result(st.session_state.history[0])
            if len(st.session_state.history) > 1:
                with st.expander("Earlier questions"):
                    for old in st.session_state.history[1:6]:
                        st.markdown(f"**Q: {old.question}**")
                        render_rag_result(old)
                        st.divider()

# --------------------------------------------------------------- compare tab
with tab_compare:
    st.subheader("Compare papers")
    st.caption(
        "Statements marked 'Stated' come from retrieved passages; 'AI interpretation' is the model's own "
        "inference. Verify numbers against the source pages."
    )
    if require_documents(2):
        picked = st.multiselect("Select two or more papers", INDEX.document_names(), key="cmp_sel")
        if st.button("Generate comparison", type="primary", disabled=len(picked) < 2):
            if llm_ready():
                with st.spinner("Gathering evidence and comparing..."):
                    st.session_state.analysis["compare"] = compare_papers(INDEX, LLM, picked, SETTINGS)
        if "compare" in st.session_state.analysis:
            render_analysis(st.session_state.analysis["compare"])

# --------------------------------------------------------------- summary tab
with tab_summary:
    st.subheader("Summarise a paper")
    st.caption("Fields the paper does not mention are reported as 'Not found' instead of being guessed.")
    if require_documents(1):
        chosen = st.selectbox("Paper", INDEX.document_names(), key="sum_sel")
        if st.button("Generate summary", type="primary"):
            if llm_ready():
                with st.spinner("Summarising..."):
                    st.session_state.analysis["summary"] = summarize_paper(INDEX, LLM, chosen, SETTINGS)
        if "summary" in st.session_state.analysis:
            render_analysis(st.session_state.analysis["summary"])

# ------------------------------------------------------------------ gaps tab
with tab_gaps:
    st.subheader("Research gap analysis (experimental)")
    st.warning(GAP_DISCLAIMER)
    if require_documents(2):
        gap_picked = st.multiselect("Select two or more papers", INDEX.document_names(), key="gap_sel")
        if st.button("Analyse gaps", type="primary", disabled=len(gap_picked) < 2):
            if llm_ready():
                with st.spinner("Analysing..."):
                    st.session_state.analysis["gaps"] = analyze_research_gaps(INDEX, LLM, gap_picked, SETTINGS)
        if "gaps" in st.session_state.analysis:
            render_analysis(st.session_state.analysis["gaps"])

# ------------------------------------------------------------- evaluation tab
with tab_eval:
    st.subheader("Evaluation dashboard")
    st.write(
        "Runs the RAG pipeline on two **synthetic** sample papers with known answers "
        f"(`{EVAL_DATASET_PATH.name}`). It uses its own temporary index, so your uploaded documents are not touched. "
        "All numbers below come from the run you trigger here."
    )
    eval_k = st.slider("K (chunks retrieved per question)", 1, 10, 3, key="eval_k")
    with_llm = st.checkbox(
        "Also evaluate generated answers (uses Groq API calls)", value=False, disabled=LLM is None
    )
    with_judge = st.checkbox("Add LLM faithfulness judge (extra API calls)", value=False, disabled=not with_llm)
    if st.button("Run evaluation", type="primary"):
        if not (SAMPLE_DOCS_DIR.exists() and any(SAMPLE_DOCS_DIR.glob("*.pdf"))):
            st.error("Sample documents are missing. Run scripts/generate_sample_documents.py.")
        else:
            with st.spinner("Indexing sample papers and evaluating..."):
                eval_index, _ = build_eval_index(load_embeddings(SETTINGS.embedding_model), SETTINGS)
                st.session_state.eval_report = run_evaluation(
                    eval_index,
                    load_dataset(),
                    SETTINGS,
                    k=eval_k,
                    llm=LLM if with_llm else None,
                    use_llm_judge=with_judge,
                    extra_info={"Embedding model": SETTINGS.embedding_model, "LLM": SETTINGS.groq_model if with_llm else "none"},
                    delay_seconds=1.0 if with_llm else 0.0,
                )
    report = st.session_state.eval_report
    if report is not None:
        aggregates = report.aggregates()
        shown = {name: value for name, value in aggregates.items() if value is not None}
        cols = st.columns(max(1, min(4, len(shown))))
        for i, (name, value) in enumerate(shown.items()):
            cols[i % len(cols)].metric(name, f"{value:.3f}")
        rows = [
            {
                "ID": r.question_id,
                "Question": r.question,
                f"P@{report.k}": r.precision_at_k,
                f"R@{report.k}": r.recall_at_k,
                "Answer F1": r.answer_f1,
                "Faithfulness (lex)": r.faithfulness_lexical,
                "Abstained": r.abstained,
            }
            for r in report.results
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.download_button("Download report (Markdown)", report.to_markdown(), file_name="researchmate_evaluation.md")
    with st.expander("Limitations of automatic evaluation"):
        for text in LIMITATIONS:
            st.markdown(f"- {text}")
    st.caption(f"Evaluation chunking: {EVAL_CHUNK_SIZE} characters, {EVAL_CHUNK_OVERLAP} overlap.")
