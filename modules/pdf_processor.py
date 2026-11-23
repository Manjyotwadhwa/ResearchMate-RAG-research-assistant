"""PDF validation and text extraction with PyMuPDF.

Design notes
- Everything is done in memory from the uploaded bytes; nothing is written to disk.
- Text is extracted block by block so paragraph boundaries survive. The text
  splitter later uses those boundaries to avoid cutting sentences in half.
- Scanned (image-only) PDFs contain no text layer. OCR is NOT supported in this
  version, so we detect them and show an explanatory error instead of failing silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from config.settings import Settings
from utils.helpers import clean_text, safe_display_name, sha256_bytes

PDF_MAGIC = b"%PDF-"

SCANNED_PDF_MESSAGE = (
    "No selectable text was found. This looks like a scanned (image-only) PDF. "
    "ResearchMate does not include OCR in this version, so scanned documents are not "
    "supported. Run the file through an OCR tool (for example OCRmyPDF) and upload the result."
)


class PDFProcessingError(Exception):
    """Raised when a file cannot be accepted or read. Message is safe to show to users."""


class EmptyPDFError(PDFProcessingError):
    """The PDF has no pages."""


class ScannedPDFError(PDFProcessingError):
    """The PDF has pages but (almost) no extractable text, so it is probably scanned."""


@dataclass
class PageText:
    page_number: int  # 1-based, matches what a reader sees
    text: str


@dataclass
class ParsedDocument:
    name: str
    doc_hash: str
    total_pages: int
    pages: list[PageText]
    empty_page_numbers: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def validate_pdf_bytes(name: str, data: bytes, max_bytes: int) -> None:
    """Cheap checks before spending time parsing: extension, size, magic bytes."""
    if not name.lower().endswith(".pdf"):
        raise PDFProcessingError(f"'{name}' is not a PDF file (only .pdf is accepted).")
    if not data:
        raise PDFProcessingError(f"'{name}' is empty (0 bytes).")
    if len(data) > max_bytes:
        raise PDFProcessingError(
            f"'{name}' is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_bytes // 1024 // 1024} MB."
        )
    # The extension can be faked, so check the file signature too.
    if not data.lstrip()[:5] == PDF_MAGIC:
        raise PDFProcessingError(f"'{name}' does not look like a valid PDF (bad file signature).")


def extract_pdf(name: str, data: bytes, settings: Settings) -> ParsedDocument:
    """Validate and extract page-wise text from PDF bytes.

    Raises PDFProcessingError (or a subclass) with a user-friendly message.
    """
    display_name = safe_display_name(name)
    validate_pdf_bytes(display_name, data, settings.max_file_size_bytes)

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for corrupt files
        raise PDFProcessingError(
            f"'{display_name}' could not be opened. The file may be corrupted."
        ) from exc

    try:
        if pdf.needs_pass:
            raise PDFProcessingError(
                f"'{display_name}' is password-protected. Remove the password and upload again."
            )
        total_pages = pdf.page_count
        if total_pages == 0:
            raise EmptyPDFError(f"'{display_name}' has no pages.")
        if total_pages > settings.max_pages_per_file:
            raise PDFProcessingError(
                f"'{display_name}' has {total_pages} pages; the limit is {settings.max_pages_per_file}."
            )

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index in range(total_pages):
            try:
                page_text = _extract_page_text(pdf[index])
            except Exception as exc:
                raise PDFProcessingError(
                    f"Text extraction failed on page {index + 1} of '{display_name}'."
                ) from exc
            if page_text:
                pages.append(PageText(page_number=index + 1, text=page_text))
            else:
                empty_pages.append(index + 1)
    finally:
        pdf.close()

    parsed = ParsedDocument(
        name=display_name,
        doc_hash=sha256_bytes(data),
        total_pages=total_pages,
        pages=pages,
        empty_page_numbers=empty_pages,
    )

    if parsed.total_chars < settings.min_text_chars:
        raise ScannedPDFError(f"'{display_name}': {SCANNED_PDF_MESSAGE}")

    if empty_pages and len(empty_pages) / total_pages > 0.3:
        parsed.warnings.append(
            f"{len(empty_pages)} of {total_pages} pages have no extractable text "
            "(possibly scanned pages or figures); they were skipped."
        )
    return parsed


def _extract_page_text(page: "pymupdf.Page") -> str:
    """Extract one page as paragraphs separated by blank lines."""
    # Each block is (x0, y0, x1, y1, text, block_no, block_type); type 0 = text.
    blocks = page.get_text("blocks", sort=True)
    paragraphs = []
    for block in blocks:
        if block[6] != 0:
            continue
        cleaned = clean_text(block[4])
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)
