"""Upload -> extract -> chunk -> embed -> index, with duplicate detection.

This sits between the Streamlit UI and the lower-level modules so the whole
ingestion workflow can be unit-tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from config.settings import Settings
from modules.pdf_processor import PDFProcessingError, extract_pdf
from modules.text_splitter import chunk_document
from modules.vector_store import DocumentIndex, DocumentInfo
from utils.helpers import safe_display_name, sha256_bytes


@dataclass
class IngestReport:
    """Outcome for one uploaded file (shown in the sidebar)."""

    name: str
    status: str  # "added" | "duplicate" | "rejected"
    message: str
    pages: int = 0
    chunks: int = 0
    doc_hash: str = ""


def _unique_name(name: str, existing: list[str]) -> str:
    """Give two different files with the same name distinct display names."""
    if name not in existing:
        return name
    path = PurePath(name)
    counter = 2
    while f"{path.stem} ({counter}){path.suffix}" in existing:
        counter += 1
    return f"{path.stem} ({counter}){path.suffix}"


def ingest_files(
    files: list[tuple[str, bytes]],
    index: DocumentIndex,
    settings: Settings,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[IngestReport]:
    """Process uploaded (filename, bytes) pairs into the index.

    Duplicates are detected by SHA-256 of the file content, so renaming a file
    does not fool the check and the same paper is never embedded twice.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    reports: list[IngestReport] = []

    for raw_name, data in files:
        name = safe_display_name(raw_name)
        doc_hash = sha256_bytes(data)
        if index.has_document(doc_hash):
            reports.append(
                IngestReport(name, "duplicate", "Already processed in this session.", doc_hash=doc_hash)
            )
            continue
        if len(index.documents) >= settings.max_files:
            reports.append(
                IngestReport(name, "rejected", f"Limit of {settings.max_files} documents reached.")
            )
            continue
        try:
            parsed = extract_pdf(_unique_name(name, index.document_names()), data, settings)
            chunks = chunk_document(parsed, size, overlap)
            if not chunks:
                raise PDFProcessingError(f"'{name}' produced no text chunks.")
            index.add_chunks(chunks)
        except PDFProcessingError as exc:
            reports.append(IngestReport(name, "rejected", str(exc)))
            continue
        except Exception as exc:  # embedding/model errors must not crash the app
            reports.append(
                IngestReport(name, "rejected", f"Unexpected error while indexing '{name}': {exc}")
            )
            continue

        index.register_document(
            DocumentInfo(
                name=parsed.name,
                doc_hash=parsed.doc_hash,
                pages=parsed.total_pages,
                chunks=len(chunks),
                chars=parsed.total_chars,
                warnings=list(parsed.warnings),
            )
        )
        note = " ".join(parsed.warnings) if parsed.warnings else "Indexed successfully."
        reports.append(
            IngestReport(
                parsed.name, "added", note, pages=parsed.total_pages, chunks=len(chunks), doc_hash=parsed.doc_hash
            )
        )
    return reports
