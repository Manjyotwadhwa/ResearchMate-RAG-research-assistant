"""Shared fixtures: synthetic PDFs built in memory and a fake offline setup."""

from __future__ import annotations

import pymupdf
import pytest

from config.settings import Settings
from modules.embeddings import HashingEmbeddings
from modules.vector_store import DocumentIndex

LONG_TEXT_A = (
    "LeafLite is a lightweight convolutional network for plant disease classification. "
    "It uses depthwise separable convolutions and was trained with the Adam optimizer. "
    "The VillageLeaf dataset contains eighteen thousand leaf images."
)
LONG_TEXT_B = (
    "CropFormer is a vision transformer pretrained with masked image modeling. "
    "Pretraining requires many GPUs and the model is too heavy for mobile devices. "
    "The FieldCrop dataset contains labeled crop disease images."
)


def make_pdf(pages: list[str], encrypt_password: str | None = None) -> bytes:
    """Create a small text PDF in memory (one string per page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 540, 780), text, fontsize=11)
    if encrypt_password:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=encrypt_password, owner_pw=encrypt_password)
    else:
        data = doc.tobytes()
    doc.close()
    return data


def make_image_only_pdf() -> bytes:
    """A PDF whose only content is an image, i.e. like a scanned page (no text layer)."""
    doc = pymupdf.open()
    page = doc.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pixmap)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def settings() -> Settings:
    # Constructed directly (not from the environment) so tests are hermetic.
    return Settings(
        groq_api_key="test-key",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        min_text_chars=50,
        max_file_size_mb=1,
        max_files=5,
    )


@pytest.fixture
def two_paper_index(settings) -> DocumentIndex:
    from modules.ingestion import ingest_files

    index = DocumentIndex(HashingEmbeddings())
    reports = ingest_files(
        [("paper_a.pdf", make_pdf([LONG_TEXT_A])), ("paper_b.pdf", make_pdf([LONG_TEXT_B]))],
        index,
        settings,
    )
    assert [r.status for r in reports] == ["added", "added"]
    return index
