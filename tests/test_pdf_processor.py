import pytest

from modules.pdf_processor import (
    EmptyPDFError,
    PDFProcessingError,
    ScannedPDFError,
    extract_pdf,
)
from tests.conftest import LONG_TEXT_A, make_image_only_pdf, make_pdf
from utils.helpers import clean_text, safe_display_name


def test_extracts_text_with_page_numbers(settings):
    data = make_pdf([LONG_TEXT_A, LONG_TEXT_A.replace("LeafLite", "SecondPage")])
    parsed = extract_pdf("paper.pdf", data, settings)
    assert parsed.name == "paper.pdf"
    assert parsed.total_pages == 2
    assert [p.page_number for p in parsed.pages] == [1, 2]
    assert "depthwise separable" in parsed.pages[0].text
    assert "SecondPage" in parsed.pages[1].text
    assert len(parsed.doc_hash) == 64


def test_same_bytes_give_same_hash(settings):
    data = make_pdf([LONG_TEXT_A])
    assert extract_pdf("a.pdf", data, settings).doc_hash == extract_pdf("b.pdf", data, settings).doc_hash


def test_rejects_non_pdf_extension(settings):
    with pytest.raises(PDFProcessingError, match="not a PDF"):
        extract_pdf("notes.txt", make_pdf([LONG_TEXT_A]), settings)


def test_rejects_fake_pdf_signature(settings):
    with pytest.raises(PDFProcessingError, match="valid PDF"):
        extract_pdf("fake.pdf", b"this is not a pdf at all", settings)


def test_rejects_empty_file(settings):
    with pytest.raises(PDFProcessingError, match="empty"):
        extract_pdf("empty.pdf", b"", settings)


def test_rejects_oversized_file(settings):
    big = b"%PDF-" + b"0" * (settings.max_file_size_bytes + 1)
    with pytest.raises(PDFProcessingError, match="limit"):
        extract_pdf("big.pdf", big, settings)


def test_corrupted_pdf_is_reported_not_crashed(settings):
    corrupted = b"%PDF-1.7\n" + b"garbage" * 50
    with pytest.raises(PDFProcessingError):
        extract_pdf("broken.pdf", corrupted, settings)


def test_blank_pdf_is_detected_as_no_text(settings):
    with pytest.raises(ScannedPDFError, match="OCR"):
        extract_pdf("blank.pdf", make_pdf([""]), settings)


def test_image_only_pdf_is_detected_as_scanned(settings):
    with pytest.raises(ScannedPDFError, match="scanned"):
        extract_pdf("scan.pdf", make_image_only_pdf(), settings)


def test_password_protected_pdf_rejected(settings):
    with pytest.raises(PDFProcessingError, match="password"):
        extract_pdf("locked.pdf", make_pdf([LONG_TEXT_A], encrypt_password="secret"), settings)


def test_partially_empty_pdf_warns(settings):
    parsed = extract_pdf("mixed.pdf", make_pdf([LONG_TEXT_A, "", ""]), settings)
    assert parsed.empty_page_numbers == [2, 3]
    assert parsed.warnings


def test_error_classes_are_subclasses():
    assert issubclass(ScannedPDFError, PDFProcessingError)
    assert issubclass(EmptyPDFError, PDFProcessingError)


def test_clean_text_normalises_pdf_artifacts():
    raw = "Deep learn-\ning is  great.\nSecond line\x00 here.\n\n\n\nNew paragraph with \ufb01nal ligature."
    cleaned = clean_text(raw)
    assert "learning" in cleaned
    assert "\x00" not in cleaned
    assert "Second line here." in cleaned
    assert "\n\n" in cleaned and "\n\n\n" not in cleaned
    assert "final" in cleaned  # ligature expanded


def test_safe_display_name_strips_paths():
    assert safe_display_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert safe_display_name("C:\\Users\\me\\paper.pdf") == "paper.pdf"
