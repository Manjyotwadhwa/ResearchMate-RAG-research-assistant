"""Split page text into overlapping chunks that keep their source metadata.

Why chunk at all?
1. Embedding models have an input limit and give blurrier vectors for long text.
2. Retrieval should return the *specific* passage that answers a question, not a
   whole paper.
3. The LLM prompt has a limited budget; small relevant chunks fit, whole papers do not.

Why split page by page? Each chunk then has an exact page number for citations.
The trade-off is that a sentence spanning a page break is split in two.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.pdf_processor import ParsedDocument


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create the splitter, validating the parameters first."""
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try paragraph breaks first, then lines, sentences, words, and finally characters.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_document(parsed: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Turn a ParsedDocument into LangChain Documents (chunks) with metadata.

    Metadata on every chunk: source (document name), page (1-based), doc_hash,
    chunk_index (position inside the page) and a unique chunk_id.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for page in parsed.pages:
        for chunk_index, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": parsed.name,
                        "page": page.page_number,
                        "doc_hash": parsed.doc_hash,
                        "chunk_index": chunk_index,
                        "chunk_id": f"{parsed.doc_hash[:8]}-p{page.page_number}-c{chunk_index}",
                    },
                )
            )
    return chunks
