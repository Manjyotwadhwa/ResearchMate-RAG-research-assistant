# ResearchMate

AI-powered multi-document research assistant. Upload several research PDFs, ask questions grounded in their text (RAG), compare papers, generate structured summaries, explore experimental research-gap hypotheses, and run a built-in evaluation.

- **Frontend:** Streamlit  **PDF parsing:** PyMuPDF  **Orchestration:** LangChain (documents, splitter, prompts, Groq client)
- **Embeddings:** Sentence Transformers (local, free)  **Vector search:** FAISS (exact, in-memory)  **LLM:** Groq (model set by env var)

> Do not upload confidential or unpublished research. Extracted passages are sent to the Groq API to generate answers.

## Verification status (read this first)

| Area | Status |
|---|---|
| All modules, Streamlit UI | **Implemented** |
| PDF extraction, scanned/corrupt/encrypted handling, chunking, FAISS index, ingestion/dedup, citation resolution, RAG flow, comparison/summary/gap flow, evaluator metrics | **Tested** by 59 pytest tests (Python 3.13, offline) using in-memory PDFs, a lexical `HashingEmbeddings` stand-in and LangChain fake chat models |
| Streamlit app | **Smoke-tested** headlessly with `streamlit.testing` (renders, metrics, chat with no API key, evaluation tab). Browser file-upload widget was *not* driven |
| Evaluator CLI | **Run** in `--offline` mode only (lexical embeddings, no LLM) |
| Real `all-MiniLM-L6-v2` embeddings | **Not tested** (model download was blocked in the build sandbox) |
| Real Groq calls, answer quality, prompt behaviour | **Not tested** (no API key / network) |
| Python 3.11 | **Not tested** (built and tested on 3.13; dependencies should support 3.11 but this is unverified) |
| Streamlit Community Cloud deployment, torch install size/memory | **Not verified** |

The evaluation scores you get from `--offline` are plumbing checks only. Run the evaluator with real embeddings and your Groq key to get meaningful numbers.

## Project structure

```
ResearchMate/
├── app.py                      Streamlit UI (wiring only)
├── config/settings.py          env-driven settings
├── modules/
│   ├── pdf_processor.py        validation + PyMuPDF extraction
│   ├── text_splitter.py        page-aware chunking with metadata
│   ├── embeddings.py           local embeddings (+ offline test embeddings)
│   ├── vector_store.py         FAISS index + metadata
│   ├── ingestion.py            upload -> index workflow, dedup
│   ├── llm_client.py           Groq client + error translation
│   ├── rag_pipeline.py         retrieve / generate / cite + shared evidence gathering
│   ├── paper_comparison.py  summarizer.py  research_gap.py
│   └── evaluator.py            metrics + CLI
├── utils/helpers.py  prompts.py
├── tests/                      59 tests
├── data/eval_dataset.json      evaluation questions
├── data/sample_documents/      2 SYNTHETIC (fictional) papers
├── scripts/generate_sample_documents.py
├── requirements.txt  requirements-dev.txt  .env.example  .gitignore
└── PROJECT_WALKTHROUGH.md      full explanation + 50+ interview questions
```

## Local setup (fresh environment)

```bash
cd ResearchMate
python3.11 -m venv .venv            # 3.11 is the target; 3.13 was used for testing
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt # use requirements.txt if you do not need tests
cp .env.example .env                # then edit .env and set GROQ_API_KEY
streamlit run app.py
```

Get a free key at the Groq console. Model names change over time: if you see a "model rejected" error, set `GROQ_MODEL` to a chat model listed in your Groq console. The first run downloads the embedding model (~90 MB) from Hugging Face.

### Run the tests

```bash
python -m pytest -q
```

### Run the evaluation (independent of the UI)

```bash
python -m modules.evaluator --offline                 # plumbing smoke test, no downloads
python -m modules.evaluator --no-llm --k 3            # real embeddings, retrieval metrics only
python -m modules.evaluator --k 3 --judge --output reports/eval.md   # + answer metrics via Groq
```

### Regenerate the sample PDFs (optional)

```bash
python scripts/generate_sample_documents.py
```

## Deploy on Streamlit Community Cloud

1. **GitHub:** create a repository, then
   ```bash
   git init && git add . && git commit -m "ResearchMate"
   git branch -M main
   git remote add origin https://github.com/<you>/ResearchMate.git
   git push -u origin main
   ```
   Check that `.env` and `.streamlit/secrets.toml` are **not** in the commit (`git status`, `.gitignore` covers both).
2. Go to share.streamlit.io, choose the repository, branch `main`, main file `app.py`. In *Advanced settings* choose Python 3.11 (or a version you have tested) and paste into **Secrets**:
   ```toml
   GROQ_API_KEY = "your_key"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
3. Deploy. The app copies `st.secrets` values into environment variables at start-up.

**Storage limits:** free cloud instances have ephemeral disk and limited memory. ResearchMate never writes uploaded PDFs or indexes to disk. Everything lives in the user's session memory and disappears when the session ends or the app restarts, so users must re-upload after a restart. The FAISS index is rebuilt per session.

**Install size caveat:** `sentence-transformers` pulls in PyTorch. A default Linux install can be large and may exceed free-tier resources. A common mitigation is the CPU-only PyTorch wheel (add `--extra-index-url https://download.pytorch.org/whl/cpu` and a CPU torch pin to `requirements.txt`). This was not tested here.

## Configuration reference

All settings are environment variables listed in `.env.example` (chunk size/overlap, top-k, upload limits, timeouts, evidence budget).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "GROQ_API_KEY is not set" | Create `.env` (local) or add Secrets (cloud); restart the app |
| "key was rejected" | Wrong or revoked key |
| "rate limit reached" | Free tier limits; wait, select fewer papers, or lower `ANALYSIS_MAX_CONTEXT_CHARS` |
| "request was rejected ... model" | Model retired or misspelled; update `GROQ_MODEL` |
| "No selectable text ... scanned" | Image-only PDF; run OCR (e.g. OCRmyPDF) first. OCR is not built in |
| "password-protected" | Remove the password and re-upload |
| Embedding model download fails | No access to huggingface.co; pre-download the model or set `EMBEDDING_MODEL` to a local path |
| Cloud app crashes at start / out of memory | torch footprint; see install-size caveat |
| Answers say "evidence not found" often | Raise TOP_K, reduce chunk size, or check that the PDF text extracted properly |
| Changed chunk size has no effect | It applies only to newly processed files; press *Clear session* and re-upload |

## Known limitations

- No OCR; scanned PDFs are rejected with an explanation.
- Chunks never cross page boundaries (exact page citations vs. split sentences at page breaks).
- Tables and figures are not parsed specially; table text may be extracted out of order.
- Retrieval is dense-only (no keyword/hybrid search, no re-ranking).
- Comparison, summary and gap analysis use a limited evidence budget per paper, so they can miss details; they say "Not found in retrieved passages" instead of guessing, but the LLM can still make mistakes. Verify numbers against the cited pages.
- Research-gap output is hypothesis generation, not verified science.
- Automatic evaluation is small and imperfect (see the evaluator docstring and dashboard).
