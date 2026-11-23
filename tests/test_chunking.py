import pytest

from modules.pdf_processor import PageText, ParsedDocument
from modules.text_splitter import build_splitter, chunk_document


def _doc(text_by_page: dict[int, str]) -> ParsedDocument:
    pages = [PageText(page_number=n, text=t) for n, t in text_by_page.items()]
    return ParsedDocument(name="paper.pdf", doc_hash="a" * 64, total_pages=len(pages), pages=pages)


LONG_PARAGRAPHS = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(8))


def test_every_chunk_keeps_document_and_page_metadata():
    chunks = chunk_document(_doc({1: LONG_PARAGRAPHS, 4: LONG_PARAGRAPHS}), 300, 50)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "paper.pdf"
        assert chunk.metadata["page"] in (1, 4)
        assert chunk.metadata["doc_hash"] == "a" * 64
    assert {c.metadata["page"] for c in chunks} == {1, 4}


def test_chunks_respect_size_limit():
    chunks = chunk_document(_doc({1: LONG_PARAGRAPHS}), 300, 50)
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_document(_doc({1: LONG_PARAGRAPHS, 2: LONG_PARAGRAPHS}), 300, 50)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_smaller_chunk_size_gives_more_chunks():
    small = chunk_document(_doc({1: LONG_PARAGRAPHS}), 200, 20)
    large = chunk_document(_doc({1: LONG_PARAGRAPHS}), 800, 20)
    assert len(small) > len(large)


def test_overlap_repeats_text_between_neighbouring_chunks():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = chunk_document(_doc({1: text}), 200, 60)
    assert len(chunks) > 2
    first_tail = chunks[0].page_content.split()[-1]
    assert first_tail in chunks[1].page_content


def test_short_page_becomes_single_chunk():
    chunks = chunk_document(_doc({1: "A short page of text."}), 1000, 200)
    assert len(chunks) == 1


@pytest.mark.parametrize("size,overlap", [(300, 300), (300, 400), (300, -1), (10, 0)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        build_splitter(size, overlap)
