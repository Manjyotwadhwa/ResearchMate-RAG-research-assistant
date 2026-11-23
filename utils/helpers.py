"""Small, dependency-free helpers shared across modules."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import PurePath

# Matches citation tags the LLM is told to emit: [S1], [E3], [S1, S2], [E1; E4]
_CITATION_RE = re.compile(r"[\[\u3010]\s*((?:[SE]\d+)(?:\s*[,;]\s*[SE]\d+)*)\s*[\]\u3011]")


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes; used to detect duplicate uploads."""
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    """Normalise text extracted from a PDF.

    Steps: Unicode NFKC (turns ligatures like 'fi' into 'fi'), drop NUL/control
    characters, re-join words hyphenated across line breaks, turn single line
    breaks into spaces (keeping blank lines as paragraph breaks) and collapse
    runs of whitespace.

    Trade-off: de-hyphenation also merges genuine hyphens that happen to fall
    at a line end ("state-\\nof-the-art" -> "stateof-the-art"). For retrieval
    this is the lesser evil compared with split words.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline -> space
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_display_name(filename: str) -> str:
    """Strip any path and control characters from an uploaded file name."""
    name = PurePath(str(filename).replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name or "unnamed.pdf"


def sanitize_question(question: str, max_chars: int = 1000) -> str:
    """Validate and normalise a user question.

    The text is only ever passed to the LLM as *data* inside a prompt, never
    executed or used to build file paths or shell commands.
    """
    if question is None:
        raise ValueError("Please enter a question.")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(question))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Please enter a question.")
    if len(cleaned) > max_chars:
        raise ValueError(f"Question is too long ({len(cleaned)} characters; limit is {max_chars}).")
    return cleaned


def format_citation(source: str, page: int) -> str:
    """Human-readable citation label, e.g. 'paper.pdf, p.3'."""
    return f"{source}, p.{page}"


def extract_cited_ids(text: str) -> list[str]:
    """Return citation IDs (e.g. 'S1', 'E4') found in LLM output, in order, unique."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        for cid in re.split(r"[,;]", match.group(1)):
            cid = cid.strip()
            if cid and cid not in seen:
                seen.append(cid)
    return seen


def resolve_citations(text: str, labels: dict[str, str]) -> tuple[str, list[str]]:
    """Replace citation tags with real document/page labels.

    ``labels`` maps an ID such as 'S2' to 'paper.pdf, p.4'. IDs that the model
    emitted but that do not exist in ``labels`` are rewritten as
    '[unverified:S9]' and returned so the UI can warn about them. This is how
    we stop the model from citing passages it was never shown.
    """
    unknown: list[str] = []

    def _replace(match: re.Match) -> str:
        parts = []
        for cid in re.split(r"[,;]", match.group(1)):
            cid = cid.strip()
            if cid in labels:
                parts.append(labels[cid])
            else:
                unknown.append(cid)
                parts.append(f"unverified:{cid}")
        return "[" + "; ".join(parts) + "]"

    resolved = _CITATION_RE.sub(_replace, text or "")
    return resolved, sorted(set(unknown))


def truncate(text: str, max_chars: int) -> str:
    """Shorten text to ``max_chars`` with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
