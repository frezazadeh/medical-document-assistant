from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status

from app.api.deps import get_container, require_api_key
from app.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentOut,
    HealthOut,
    UploadOut,
)
from app.container import Container
from app.ingestion.loaders import EmptyDocument, UnsupportedDocument

public = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])

SNIPPET_CHARS = 320


@public.get("/health", response_model=HealthOut, tags=["meta"])
def health(container: Container = Depends(get_container)):
    settings = container.settings
    return HealthOut(
        status="ok",
        llm_provider=container.llm.provider,
        llm_model=container.llm.model,
        embedding_model=settings.embedding_model,
        documents=len(container.registry.list()),
    )


@protected.post(
    "/documents", response_model=UploadOut, status_code=status.HTTP_201_CREATED, tags=["documents"]
)
def upload_document(
    file: UploadFile, response: Response, container: Container = Depends(get_container)
):
    limit = container.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File is larger than {container.settings.max_upload_mb} MB.",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")

    try:
        record, created = container.ingestion.ingest(file.filename or "upload.txt", data)
    except UnsupportedDocument as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except EmptyDocument as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    if not created:
        response.status_code = status.HTTP_200_OK
    return UploadOut(**_document_fields(record), created=created)


@protected.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(container: Container = Depends(get_container)):
    return [DocumentOut(**_document_fields(r)) for r in container.registry.list()]


@protected.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
def delete_document(document_id: str, container: Container = Depends(get_container)):
    if not container.ingestion.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")


@protected.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: AskRequest, container: Container = Depends(get_container)):
    if request.document_ids:
        unknown = [d for d in request.document_ids if container.registry.get(d) is None]
        if unknown:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown document ids: {unknown}")

    result = container.qa.ask(
        question=request.question,
        document_ids=request.document_ids,
        history=[turn.model_dump() for turn in request.history],
    )
    return AskResponse(
        answer=result.answer,
        citations=[
            Citation(
                ref=s.ref,
                document_id=s.document_id,
                filename=s.filename,
                page=s.page,
                snippet=_snippet(s.text),
                score=s.score,
            )
            for s in result.citations
        ],
        grounded=result.grounded,
        unsupported=result.unsupported,
        trace_id=result.trace_id,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@protected.get("/traces", tags=["observability"])
def recent_traces(limit: int = 20, container: Container = Depends(get_container)):
    return container.traces.recent(min(limit, 100))


@protected.get("/traces/{trace_id}", tags=["observability"])
def get_trace(trace_id: str, container: Container = Depends(get_container)):
    trace = container.traces.get(trace_id)
    if trace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trace not found.")
    return trace


def _document_fields(record) -> dict:
    return {
        "id": record.id,
        "filename": record.filename,
        "pages": record.pages,
        "chunks": record.chunks,
        "uploaded_at": record.uploaded_at,
    }


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
