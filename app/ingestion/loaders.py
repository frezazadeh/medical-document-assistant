"""Turn an uploaded file into a list of pages of plain text."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UnsupportedDocument(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a reader would cite it
    text: str


def extract_pages(filename: str, data: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf(data)
    elif suffix in {".txt", ".md"}:
        pages = _load_text(data)
    else:
        raise UnsupportedDocument(f"Unsupported file type '{suffix}'. Use PDF or plain text.")

    pages = [p for p in pages if p.text]
    if not pages:
        # Typically a scanned PDF. There is no OCR step in this prototype.
        raise EmptyDocument("No extractable text found in the document.")
    return pages


def _load_pdf(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [
            Page(number=i, text=_clean(page.extract_text() or ""))
            for i, page in enumerate(reader.pages, start=1)
        ]
    except PdfReadError as exc:
        raise UnsupportedDocument(f"Could not read PDF: {exc}") from exc


def _load_text(data: bytes) -> list[Page]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    # Form feeds are the closest thing plain text has to page breaks.
    parts = text.split("\f")
    return [Page(number=i, text=_clean(part)) for i, part in enumerate(parts, start=1)]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Words hyphenated across a line break: "hyper-\ntension" -> "hypertension"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
