"""Model Context Protocol server over the same services the HTTP API uses.

The HTTP API is for people and UIs. This is the entry point for other models
and agents: an MCP client (Claude Desktop, an IDE, an orchestrating agent) can
discover what context this system offers and pull it in when it needs it.

    tools      search_documents, ask_documents, list_documents
    resources  document://{id}   full extracted text, page by page
    prompts    grounded_clinical_qa   the same system prompt the API uses

Run with:  python -m app.mcp_server   (stdio transport)
"""

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from app.config import get_settings
from app.container import Container, build_container
from app.observability.logging import configure_logging

mcp = MCPServer(
    name="medical-document-assistant",
    instructions=(
        "Search and question clinical documents that were uploaded to the Medical "
        "Document Assistant. Results carry the file name and page so they can be cited."
    ),
)


@lru_cache
def _container() -> Container:
    # Built on first use, so the client's initialize handshake isn't held up
    # by loading the embedding model.
    return build_container(get_settings())


@mcp.tool()
def list_documents() -> list[dict]:
    """List the documents that are available, with their ids and page counts."""
    return [
        {"id": d.id, "filename": d.filename, "pages": d.pages, "uploaded_at": d.uploaded_at}
        for d in _container().registry.list()
    ]


@mcp.tool()
def search_documents(query: str, top_k: int = 5, document_id: str | None = None) -> list[dict]:
    """Search the uploaded documents by meaning, not just keywords.

    Returns the most relevant passages with file name, page number and a
    similarity score between 0 and 1. Use this when you want to read the
    evidence yourself. Pass document_id to search a single document.
    """
    hits = _container().retriever.search(
        query,
        top_k=max(1, min(top_k, 20)),
        document_ids=[document_id] if document_id else None,
    )
    return [
        {
            "document_id": h.document_id,
            "filename": h.filename,
            "page": h.page,
            "score": h.score,
            "text": h.text,
        }
        for h in hits
    ]


@mcp.tool()
def ask_documents(question: str, document_id: str | None = None) -> dict:
    """Answer a question from the uploaded documents, with citations.

    Runs the full retrieval and answering pipeline. Use this when you want a
    finished, cited answer rather than raw passages.
    """
    result = _container().qa.ask(question, document_ids=[document_id] if document_id else None)
    return {
        "answer": result.answer,
        "grounded": result.grounded,
        "unsupported": result.unsupported,
        "trace_id": result.trace_id,
        "citations": [
            {"ref": s.ref, "filename": s.filename, "page": s.page, "text": s.text}
            for s in result.citations
        ],
    }


@mcp.resource("document://{document_id}", mime_type="text/plain")
def document_text(document_id: str) -> str:
    """Full extracted text of one document, with page markers."""
    container = _container()
    record = container.registry.get(document_id)
    if record is None:
        raise ValueError(f"Unknown document id: {document_id}")
    pages = container.registry.read_pages(document_id)
    body = "\n\n".join(f"[page {p['page']}]\n{p['text']}" for p in pages)
    return f"{record.filename}\n\n{body}"


@mcp.prompt()
def grounded_clinical_qa(question: str) -> str:
    """Instructions for answering a clinical question strictly from retrieved sources."""
    system = _container().context_builder.system_prompt()
    return (
        f"{system}\n\n"
        "Use the search_documents tool to collect sources, number them in the order "
        f"you use them, then answer.\n\nQuestion: {question}"
    )


def main() -> None:
    configure_logging(get_settings().log_level)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
