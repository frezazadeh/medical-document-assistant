import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import extract_pages
from app.retrieval.vector_store import VectorStore
from app.storage import DocumentRecord, DocumentRegistry

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        registry: DocumentRegistry,
        store: VectorStore,
        uploads_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._registry = registry
        self._store = store
        self._uploads_dir = uploads_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, filename: str, data: bytes) -> tuple[DocumentRecord, bool]:
        """Returns the document record and whether it was newly created."""
        started = time.perf_counter()
        sha256 = hashlib.sha256(data).hexdigest()

        existing = self._registry.find_by_hash(sha256)
        if existing:
            logger.info("document already ingested", extra={"document_id": existing.id})
            return existing, False

        filename = Path(filename).name  # never trust a client-supplied path
        pages = extract_pages(filename, data)
        chunks = chunk_pages(pages, self._chunk_size, self._chunk_overlap)

        record = DocumentRecord(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            sha256=sha256,
            pages=len(pages),
            chunks=len(chunks),
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        (self._uploads_dir / f"{record.id}{Path(filename).suffix.lower()}").write_bytes(data)
        self._store.add(record.id, filename, chunks)
        # Registered last, so a failed embedding run doesn't leave a document
        # that looks ingested but has no chunks behind it.
        self._registry.add(record, pages)

        logger.info(
            "document ingested",
            extra={
                "document_id": record.id,
                "doc_filename": filename,
                "pages": record.pages,
                "chunks": record.chunks,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return record, True

    def delete(self, document_id: str) -> bool:
        record = self._registry.get(document_id)
        if record is None:
            return False
        self._store.delete_document(document_id)
        self._registry.delete(document_id)
        for path in self._uploads_dir.glob(f"{document_id}.*"):
            path.unlink()
        logger.info("document deleted", extra={"document_id": document_id})
        return True
