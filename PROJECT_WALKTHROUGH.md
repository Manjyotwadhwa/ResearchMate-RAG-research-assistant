# ResearchMate - Project Walkthrough

This document explains the implementation as it actually exists in the repository. Where something was not verified in the build environment, it says so (see section 12).

---

## 1. Architecture

```
                        ┌──────────────────────────────────────────┐
  PDF uploads ─────────▶│ app.py (Streamlit UI, session_state)     │
                        └───────┬───────────────────────┬──────────┘
                                │ ingest                 │ ask / compare / summarise / gaps / evaluate
                                ▼                        ▼
                   modules/ingestion.py          modules/rag_pipeline.py ──▶ modules/llm_client.py ──▶ Groq API
                     │        │       │             │   (retrieve → generate)
                     ▼        ▼       ▼             ▼
          pdf_processor  text_splitter  vector_store.DocumentIndex ◀── embeddings.py (local MiniLM)
          (PyMuPDF)      (LangChain)    (FAISS IndexFlatIP + metadata)
                                                    ▲
                           evaluator.py ────────────┘ (own index built from data/sample_documents)
```

Layers: **UI** (`app.py`) → **workflows** (`ingestion`, `rag_pipeline`, feature modules) → **building blocks** (`pdf_processor`, `text_splitter`, `embeddings`, `vector_store`, `llm_client`) → **config/utils**. Dependencies only point downward, which is what makes the modules testable without Streamlit or the network.

Deliberate design decisions:
- **Everything is in-memory and per session.** No uploaded document touches disk; the index is rebuilt each session. This fits Streamlit Cloud's ephemeral storage and avoids retaining private documents.
- **Retrieval and generation are separate methods** (`RAGPipeline.retrieve`, `RAGPipeline.generate`) so retrieval can be evaluated without an LLM.
- **Embeddings and the LLM are injected**, so tests use `HashingEmbeddings` and LangChain fake chat models.
- **FAISS is used directly**, not through `langchain-community`, which is being sunset. LangChain is still used for `Document`, `RecursiveCharacterTextSplitter`, prompt templates, `Embeddings` interface and `ChatGroq`.

## 2. End-to-end data flow

**Ingestion** (sidebar upload):
1. Streamlit gives `(filename, bytes)` for each new file (already-seen `file_id`s are skipped).
2. `ingest_files` computes a SHA-256 of the bytes; if the hash is already in the registry the file is reported as a duplicate.
3. `extract_pdf` validates extension, size and `%PDF-` signature, opens the bytes with PyMuPDF, rejects encrypted/over-long files, extracts text per page as blocks, cleans it, and raises `ScannedPDFError` if almost no text exists.
4. `chunk_document` splits each page with `RecursiveCharacterTextSplitter` and attaches metadata (`source`, `page`, `doc_hash`, `chunk_index`, `chunk_id`).
5. `DocumentIndex.add_chunks` embeds all chunks (`embed_documents`), adds the vectors to a FAISS `IndexFlatIP`, and appends the Documents to a parallel list.
6. The document is registered (name, pages, chunks, chars, warnings) and shown in the sidebar.

**Question answering:**
1. `sanitize_question` validates the text. 2. `retrieve` embeds the query and searches FAISS (optionally restricted to selected documents). 3. `format_context` numbers chunks `S1..Sn`. 4. `RAG_PROMPT` is filled with context and question. 5. `call_llm` invokes Groq and translates failures. 6. `resolve_citations` rewrites `[S1]` to `[paper.pdf, p.3]`, flags unknown tags. 7. The UI shows answer, source list (marking cited ones) and an expander with the retrieved passages and scores.

**Comparison / summary / gaps:** for each selected paper `gather_evidence` runs one retrieval per aspect (restricted to that paper), merges results round-robin under a character budget, numbers them `E1..En`, and one LLM call produces the structured text, whose citations are verified the same way.

**Evaluation:** `build_eval_index` indexes the sample PDFs into a separate index (300-char chunks); `run_evaluation` retrieves for every dataset question, scores retrieval against evidence phrases, and, if an LLM is supplied, generates answers and scores them.

## 3. Modules and their key functions

| Module | Responsibility | Key functions / classes |
|---|---|---|
| `config/settings.py` | Frozen `Settings` dataclass from env/.env; validation | `Settings.from_env`, `validate`, `get_settings` |
| `utils/helpers.py` | Pure helpers | `clean_text`, `safe_display_name`, `sanitize_question`, `sha256_bytes`, `resolve_citations`, `extract_cited_ids` |
| `utils/prompts.py` | All prompts and retrieval query lists | `RAG_PROMPT`, `COMPARISON_PROMPT`, `SUMMARY_PROMPT`, `GAP_PROMPT`, `FAITHFULNESS_JUDGE_PROMPT`, `NO_EVIDENCE_MESSAGE` |
| `modules/pdf_processor.py` | Validate + extract | `validate_pdf_bytes`, `extract_pdf`, `_extract_page_text`; errors `PDFProcessingError`, `ScannedPDFError`, `EmptyPDFError` |
| `modules/text_splitter.py` | Chunking | `build_splitter` (parameter validation), `chunk_document` |
| `modules/embeddings.py` | Embedding backends | `get_embeddings` (cached MiniLM via `HuggingFaceEmbeddings`, normalised), `HashingEmbeddings` (offline) |
| `modules/vector_store.py` | FAISS wrapper | `DocumentIndex.add_chunks`, `search` (top-k, source filter, min score), `RetrievedChunk`, `DocumentInfo` |
| `modules/ingestion.py` | Workflow + dedup + limits | `ingest_files`, `IngestReport`, `_unique_name` |
| `modules/llm_client.py` | Groq model + errors | `get_llm`, `call_llm`, `_friendly_error`, `LLMError` |
| `modules/rag_pipeline.py` | RAG + shared evidence engine | `RAGPipeline.retrieve/generate/answer`, `format_context`, `gather_evidence`, `run_document_analysis` |
| `modules/paper_comparison.py`, `summarizer.py`, `research_gap.py` | Feature entry points (thin) | `compare_papers`, `summarize_paper`, `analyze_research_gaps` |
| `modules/evaluator.py` | Metrics + CLI | `precision_at_k`, `recall_at_k`, `token_f1`, `lexical_faithfulness`, `llm_faithfulness`, `run_evaluation`, `main` |
| `app.py` | UI | session init, sidebar, five tabs, renderers |

## 4. Libraries and imports

- **streamlit** - UI, `session_state`, `cache_resource` (loads the embedding model/LLM client once per server process), `testing.AppTest` for smoke tests.
- **pymupdf** (`import pymupdf`) - fast, accurate text extraction with block layout, in-memory open from bytes.
- **langchain-core** - `Document`, `Embeddings` interface, `ChatPromptTemplate`, base chat model type, fake chat models for tests.
- **langchain-text-splitters** - `RecursiveCharacterTextSplitter`.
- **langchain-groq** - `ChatGroq` client (timeout, retries).
- **langchain-huggingface** + **sentence-transformers** - local embedding model wrapper and the model runtime (brings PyTorch).
- **faiss-cpu** - vector similarity search.
- **numpy** - vector arrays/norms. **pandas** - results table in the dashboard.
- **python-dotenv** - loads `.env`. **pytest** - tests. **reportlab** - only to generate the synthetic sample PDFs.
- Stdlib: `dataclasses`, `hashlib`, `re`, `unicodedata`, `argparse`, `json`, `pathlib`.

## 5. Why these technologies, and the alternatives

| Choice | Why | Alternatives and trade-offs |
|---|---|---|
| Streamlit | Fastest path to an interactive Python UI; free hosting on Community Cloud | Gradio (similar, ML-demo focus); FastAPI + React (more control, much more work) |
| PyMuPDF | Fast, block-level layout info, in-memory | pypdf (pure Python, weaker layout); pdfplumber (good tables, slower); Unstructured (heavy) |
| RecursiveCharacterTextSplitter | Prefers paragraph → line → sentence → word boundaries | Fixed-size (cuts sentences); semantic chunking (costly); markdown/section-aware (needs structure) |
| all-MiniLM-L6-v2 (default) | Small (~90 MB), CPU-friendly, 384-dim, free | bge-small / e5 (often better retrieval, larger or need prefixes); OpenAI embeddings (paid, data leaves the machine) |
| FAISS `IndexFlatIP` | Exact search, no tuning, trivial at thousands of vectors | Chroma/Qdrant (persistence, filtering, a server/DB); HNSW/IVF (faster at millions, approximate) |
| Groq | Free tier, very fast inference, OpenAI-style chat API | OpenAI/Anthropic (paid); Ollama (local, needs a machine that can run the model) |
| Custom evaluator | Transparent, no extra dependency, runnable offline | RAGAS (richer LLM-based metrics, but needs an LLM judge, heavier dependencies, and its API changes often) |

## 6. Chunking, embeddings, FAISS and retrieval

**Chunking.** Embedding models have an input limit (MiniLM-L6 truncates at roughly 256 word-pieces per its model card; not verified here), long text yields blurry vectors, and the LLM prompt has a budget. So each page is split into ~1000-character chunks with 200 overlap (configurable). Overlap stops a fact that sits on a boundary from being lost. Splitting per page gives exact page citations at the cost of splitting sentences that cross a page break.

**Embeddings.** An embedding model maps text to a vector so that texts with similar meaning are close. We normalise vectors to length 1, so the dot product equals cosine similarity (1 = same direction, ~0 = unrelated).

**FAISS.** FAISS stores vectors and answers "which stored vectors have the highest dot product with this query vector?". `IndexFlatIP` compares the query with every vector (exact, O(N·d)), which is milliseconds for a few thousand chunks. FAISS returns integer positions, so `DocumentIndex` keeps a parallel list of `Document`s; position *i* in FAISS ↔ `documents[i]`. This is how metadata survives retrieval.

**Filtering.** FAISS cannot filter by metadata, so with a source filter the code asks for all vectors and filters in Python (fine at this scale; a vector DB with native filtering would be the upgrade).

**Retrieval.** `search` embeds the query, takes the top-k, optionally drops results under `min_score`, and returns `RetrievedChunk(document, score)`.

## 7. The RAG workflow and prompt construction

Context injection: each retrieved chunk becomes a block `[S1] (document: X, page N)\n<text>`; the blocks are joined and substituted into `RAG_HUMAN` as `{context}` next to `{question}`. The system prompt (`RAG_SYSTEM`) tells the model to use only the passages, cite tags, answer with a fixed sentence if evidence is insufficient (`NO_EVIDENCE_MESSAGE`), never invent numbers, and treat passage text as data (a mitigation for prompt injection inside PDFs; it reduces but does not eliminate the risk). Temperature defaults to 0.1 for stable, literal answers. If retrieval returns nothing, the pipeline returns the "not found" message **without calling the LLM**.

The analysis prompts additionally require two labels: **Stated** (explicit in evidence, cited) vs **AI interpretation** (inference), and the exact phrase "Not found in retrieved passages" for missing items.

## 8. How citations are maintained

1. Chunk metadata carries `source` and `page` from extraction onward.
2. The model only ever sees short tags (`S1`, `E4`); it never writes file names or page numbers itself.
3. `resolve_citations` maps tags back to `file.pdf, p.N` using a dictionary built from the chunks actually retrieved.
4. Tags that are not in the dictionary become `[unverified:S9]` and the UI warns. So a citation can never point to a passage that was not retrieved, but note that a *valid* tag can still be attached to a claim the passage does not support; that is a faithfulness problem, not a citation-existence problem.
5. The UI also lists all retrieved sources and marks which ones the model cited.

## 9. Evaluation metrics

Ground truth for retrieval is a list of **evidence phrases** per question; a chunk is relevant if it contains one.

- **Precision@K** = relevant chunks in top-K ÷ K. With one relevant chunk per question and K=3, the maximum is 1/3, so interpret it relative to that ceiling.
- **Recall@K** = evidence phrases found in at least one retrieved chunk ÷ total phrases.
- **Hit rate@K** = questions with ≥1 relevant chunk in the top-K.
- **Mean retrieval score** = average cosine similarity of retrieved chunks (a context-relevance proxy).
- **Answer F1** = token-overlap F1 vs. expected answer; **answer similarity** = embedding cosine vs. expected answer.
- **Faithfulness (lexical)** = share of answer sentences whose numbers all appear in the context and whose content words are ≥70% present; **(LLM judge)** = optional verdict SUPPORTED/PARTIAL/UNSUPPORTED → 1/0.5/0.
- **Abstention accuracy** = for the unanswerable question, whether the system said evidence was missing.

Scores are computed only from an actual run; with no LLM, generation metrics are `n/a`, not zero. Limitations are listed in `LIMITATIONS` in `evaluator.py` and shown in the dashboard: tiny dataset, phrase-based relevance, lexical metrics reward overlap not correctness, LLM judge is fallible, offline mode is not semantic.

## 10. Common failure scenarios and debugging

| Scenario | How to diagnose |
|---|---|
| Wrong/irrelevant answers | Open *Retrieved passages*: if the right passage is absent, it is a retrieval problem (raise top-k, change chunk size, check extraction). If present but ignored, it is a generation/prompt problem |
| "Evidence not found" on answerable questions | Same check; also inspect extracted text for garbled columns/tables |
| Garbled text | Multi-column layouts or ligature fonts; inspect `ParsedDocument.pages[i].text` in a Python shell |
| Missing pages | `empty_page_numbers` / sidebar warning (scanned or figure pages) |
| API errors | `_friendly_error` classifies by exception class name: auth, rate limit, timeout, connection, bad request |
| Duplicate/odd names | Hash-based dedup; same name with different content becomes `name (2).pdf` |
| Everything slow on first run | Embedding model download and load; later runs use `cache_resource` |
| Cloud restarts | Session memory is lost by design |

## 11. Security and privacy

- API key only from environment/secrets; `.env` and `secrets.toml` are git-ignored; never logged or shown.
- File validation: extension, size, magic bytes, page count, encryption; corrupt files caught.
- No disk persistence of uploads; no shell execution; user text is only used as prompt data (sanitised, length-limited).
- Display names are stripped of paths (`safe_display_name`).
- **Privacy:** document text leaves the machine to Groq for generation. The UI warns users not to upload confidential material. Embeddings are local.
- **Prompt injection:** a malicious PDF could contain instructions. Mitigated by "treat as data" rules and verified citations; not eliminated. The app has no tools or actions, which limits the blast radius to misleading text.
- Streamlit's uploader size is also capped by `server.maxUploadSize` in `.streamlit/config.toml`.

## 12. What was and was not verified

Tested: 59 offline unit tests (extraction edge cases, chunking, FAISS index, ingestion/dedup, RAG with fake LLMs including failures, citations, comparison/summary/gap plumbing, evaluator metrics and CLI); headless Streamlit smoke test; `pyflakes` clean; all modules import.

Not verified: real MiniLM embeddings, real Groq calls and answer quality, Python 3.11, browser upload, Streamlit Cloud deployment and its memory limits. Treat these as "implemented, untested".

---

## 13. Interview questions and answers (based on this implementation)

**Fundamentals**

1. **What problem does ResearchMate solve?** It lets a user ask questions across several PDFs and get answers grounded in, and cited to, specific document pages, plus structured comparison/summaries, instead of reading every paper.
2. **What is RAG and why use it instead of just asking the LLM?** RAG retrieves relevant text first and gives it to the LLM as context. It grounds answers in the user's documents, works on content the model never saw, and makes answers checkable via citations.
3. **Walk me through a question end to end.** Sanitise → embed query → FAISS top-k → number chunks S1..Sn → fill prompt → Groq call → resolve citations → show answer, sources and retrieved passages.
4. **Why are retrieval and generation separate methods?** So retrieval can be tested/evaluated without an LLM, failures can be attributed to the right stage, and retrieved passages can still be shown if the LLM fails.
5. **What does the app do when documents don't contain the answer?** The prompt instructs a fixed "not found" sentence, which sets `insufficient_evidence`; if retrieval returns nothing at all the LLM is not called.

**PDF processing and chunking**

6. **Why PyMuPDF?** Fast, extracts layout blocks, opens from bytes in memory so nothing hits disk.
7. **Why extract by blocks?** Blocks approximate paragraphs; joining them with blank lines lets the splitter break at paragraph boundaries.
8. **How are scanned PDFs handled?** If extracted characters fall below `min_text_chars` the file raises `ScannedPDFError` with an explanation that OCR is not supported in this version.
9. **What does `clean_text` do and what is its downside?** NFKC normalisation (ligatures), control-char removal, de-hyphenation, newline-to-space, whitespace collapse. Downside: it can merge real hyphenated words split across lines.
10. **Why chunk documents?** Embedding input limits, sharper vectors, and prompt budget; retrieval returns the exact passage rather than a whole paper.
11. **Why 1000/200 defaults?** A reasonable balance for ~250-token chunks that fit MiniLM's input limit with some overlap; they are tunable, and I did not run a tuning experiment.
12. **What does overlap do?** Prevents facts at chunk boundaries from being cut off from their context; it costs extra storage and some duplicate retrieval.
13. **Why split page by page?** Exact page metadata for citations. Trade-off: sentences crossing a page break are split.
14. **How does RecursiveCharacterTextSplitter work?** It tries separators in order (`\n\n`, `\n`, `. `, space, empty), splitting on the first that yields pieces under the limit and merging pieces up to `chunk_size` with overlap.
15. **How do you avoid reprocessing the same file?** SHA-256 of the bytes stored in the registry; renamed copies are detected, and Streamlit `file_id`s prevent re-work on reruns.

**Embeddings and FAISS**

16. **What is an embedding?** A fixed-length vector (384 dims for MiniLM) representing meaning so similar texts are close.
17. **Why normalise embeddings?** Then dot product equals cosine similarity, so `IndexFlatIP` gives cosine ranking.
18. **What is cosine similarity?** The cosine of the angle between two vectors; 1 same direction, 0 orthogonal.
19. **Why FAISS `IndexFlatIP`?** Exact search with no tuning; at a few thousand chunks brute force is milliseconds. Approximate indexes help only at much larger scale.
20. **How is metadata tied to vectors?** FAISS returns positions; a parallel `_documents` list maps position to the `Document` with metadata.
21. **Why not LangChain's FAISS wrapper?** It lives in `langchain-community`, which is being sunset; the direct wrapper is ~100 lines and easier to explain.
22. **How does per-document filtering work?** FAISS can't filter, so we fetch all results for the query and filter by `source` in Python; fine at this scale, a vector DB would filter natively.
23. **Why local embeddings?** Free, private (document text isn't sent for embedding), no per-call cost. Cost: model download and CPU compute.
24. **What would you change for 1M chunks?** Persistent vector DB or FAISS IVF/HNSW, batch embedding, metadata filtering, maybe hybrid search.
25. **What is the dimension check in `add_chunks` for?** Guards against mixing vectors from different embedding models in one index.

**RAG, prompts, LLM**

26. **How is the prompt built?** System rules + human message containing numbered, labelled passages and the question (`format_context`, `RAG_PROMPT`).
27. **How do you reduce hallucinations?** Context-only instruction, fixed abstention sentence, low temperature, exact-number rule, verified citations, and showing the retrieved passages. It reduces, not removes, hallucination.
28. **Why temperature 0.1?** Near-deterministic, literal answers for factual extraction; configurable.
29. **How do you handle LLM failures?** `call_llm` wraps all exceptions into `LLMError` with friendly messages (auth, rate limit, timeout, connection, bad request); the UI still shows retrieved passages.
30. **How are timeouts/retries set?** `ChatGroq(timeout=..., max_retries=...)` from settings.
31. **What is prompt injection and how do you mitigate it?** Instructions hidden in a document that try to steer the model. The prompt says passages are data; citations are verified; the app has no tools. It's not a complete defence.
32. **Why is the LLM given tags instead of file names?** So it can't invent a filename or page; we map tags to real metadata ourselves.
33. **How do comparison/summary work with limited context?** `gather_evidence` runs one query per aspect per paper, interleaves results under a character budget, and the LLM works only from that evidence.
34. **What is the "Stated vs AI interpretation" split for?** To let users distinguish text-supported claims from model inference.
35. **Why is research-gap analysis labelled experimental?** An LLM suggesting gaps from a few excerpts can't verify novelty; outputs are hypotheses and the prompt and UI say so.
36. **Why is `min_relevance_score` 0 by default?** I had no tuned threshold; an arbitrary one could drop good chunks. It's available to configure once measured.

**Evaluation**

37. **How do you evaluate retrieval without an LLM?** Evidence phrases define relevant chunks; compute Precision@K, Recall@K and hit rate.
38. **Define Precision@K and Recall@K as implemented.** Precision@K = relevant retrieved ÷ K; Recall@K = evidence phrases covered by retrieved chunks ÷ total phrases.
39. **Why can Precision@3 be low even for a perfect system?** Each question has about one relevant chunk, so at K=3 the ceiling is ~0.33.
40. **How is faithfulness measured?** Lexical: sentence-level overlap of content words and exact numbers with the context; optional LLM judge. Lexical misses negation/logic errors.
41. **Why not RAGAS?** I chose a transparent, dependency-light module that runs offline; RAGAS offers richer judged metrics but needs an LLM judge and has a changing API. Adding it later is straightforward.
42. **What is abstention accuracy?** For unanswerable questions, whether the system says the evidence is missing.
43. **What are limitations of your evaluation?** Tiny synthetic dataset, phrase-based ground truth, overlap metrics, fallible judge, and offline mode isn't semantic. Scores indicate plumbing/regressions, not real-world quality.
44. **How do you prevent fabricated scores?** Everything is computed in `run_evaluation`; missing metrics are `n/a`.
45. **Why use smaller chunks for evaluation?** The sample corpus is tiny; 300-character chunks yield enough chunks for Precision@K to be informative.

**Engineering, deployment, security**

46. **How is the code tested without network or API keys?** Dependency injection: `HashingEmbeddings` and LangChain fake chat models; PDFs are generated in memory with PyMuPDF.
47. **What does `HashingEmbeddings` prove and not prove?** It proves index mechanics, metadata, filtering and plumbing; not semantic retrieval quality.
48. **How do you keep API keys safe?** Environment variables / Streamlit secrets only; `.env` is git-ignored; keys never rendered or logged.
49. **Why does the app not persist files or indexes?** Privacy and Streamlit Cloud's ephemeral storage; the trade-off is re-uploading after a restart.
50. **How does the app avoid reprocessing on Streamlit reruns?** Tracks processed `file_id`s in `session_state` and also dedups by content hash.
51. **What are the deployment risks?** PyTorch install size and memory on free tiers, Hugging Face download at first start, Groq rate limits, Python-version compatibility (unverified here).
52. **What would you add next?** OCR, hybrid BM25+dense retrieval, a re-ranker, section-aware chunking, table extraction, persistent per-user storage with auth, and a larger human-labelled evaluation set.
53. **How would you detect a retrieval vs generation bug in production?** Log the retrieved chunk IDs and scores with the answer; if the gold passage isn't in the chunks it's retrieval, otherwise generation.
54. **What happens if two different PDFs have the same filename?** Different hashes → the second is renamed `name (2).pdf` so citations stay unambiguous.
