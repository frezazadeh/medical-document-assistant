import pytest
from fpdf import FPDF

from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import EmptyDocument, Page, UnsupportedDocument, extract_pages


def make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def test_pdf_pages_keep_their_numbers():
    pages = extract_pages("report.pdf", make_pdf(["Dose was 5 mg daily.", "No deaths occurred."]))

    assert [p.number for p in pages] == [1, 2]
    assert "5 mg" in pages[0].text
    assert "No deaths" in pages[1].text


def test_text_file_splits_on_form_feed():
    pages = extract_pages("notes.txt", b"first page\fsecond page")

    assert [(p.number, p.text) for p in pages] == [(1, "first page"), (2, "second page")]


def test_hyphenated_line_breaks_are_joined():
    pages = extract_pages("notes.txt", b"uncontrolled hyper-\ntension in adults")

    assert "hypertension" in pages[0].text


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedDocument):
        extract_pages("scan.png", b"\x89PNG")


def test_document_without_text_is_rejected():
    with pytest.raises(EmptyDocument):
        extract_pages("empty.txt", b"   \n  ")


def test_chunks_respect_size_and_never_cross_pages():
    sentence = "Hyperkalemia was reported in ten participants. "
    pages = [Page(1, sentence * 12), Page(2, sentence * 3)]

    chunks = chunk_pages(pages, size=200, overlap=50)

    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].page == 2


def test_consecutive_chunks_overlap_by_whole_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))

    chunks = chunk_pages([Page(1, text)], size=120, overlap=40)

    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text


def test_sentence_longer_than_a_chunk_is_wrapped_not_dropped():
    words = ["token"] * 100
    chunks = chunk_pages([Page(1, " ".join(words))], size=80, overlap=10)

    assert all(len(c.text) <= 80 for c in chunks)
    assert sum(c.text.count("token") for c in chunks) >= 100


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], size=100, overlap=100)
