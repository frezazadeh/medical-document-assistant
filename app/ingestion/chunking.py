"""Sentence-aware chunking that keeps track of the page each chunk came from.

Chunks never span two pages. That costs a little recall when a sentence runs
across a page break, but it means a citation can always name one exact page.
"""

import re
from dataclasses import dataclass

from app.ingestion.loaders import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Chunk:
    index: int
    page: int
    text: str


def chunk_pages(pages: list[Page], size: int = 900, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    for page in pages:
        for text in _pack(_split_units(page.text, size), size, overlap):
            chunks.append(Chunk(index=len(chunks), page=page.number, text=text))
    return chunks


def _split_units(text: str, size: int) -> list[str]:
    """Split into sentences; hard-wrap anything that is still longer than a chunk."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > size:
                cut = sentence.rfind(" ", 0, size)
                cut = cut if cut > 0 else size
                units.append(sentence[:cut])
                sentence = sentence[cut:].lstrip()
            if sentence:
                units.append(sentence)
    return units


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        if current and length + len(unit) + 1 > size:
            chunks.append(" ".join(current))
            current = _tail(current, overlap)
            length = sum(len(u) + 1 for u in current)
        current.append(unit)
        length += len(unit) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail(units: list[str], overlap: int) -> list[str]:
    """Trailing sentences of the previous chunk that fit in the overlap budget."""
    tail: list[str] = []
    used = 0
    for unit in reversed(units):
        if used + len(unit) + 1 > overlap:
            break
        tail.insert(0, unit)
        used += len(unit) + 1
    return tail
