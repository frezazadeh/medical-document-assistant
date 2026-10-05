"""Document registry: which files were ingested, and their extracted text.

SQLite for the metadata because it is one file, needs no server and gives us
a uniqueness constraint on the content hash. The vector index only holds chunks.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.loaders import Page


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    sha256: str
    pages: int
    chunks: int
    uploaded_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentRegistry:
    def __init__(self, db_path: Path, extracted_dir: Path):
        self._db_path = db_path
        self._extracted_dir = extracted_dir
        db_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self):
        # One short-lived connection per call: FastAPI runs sync endpoints in a
        # thread pool and sqlite connections don't like being shared across threads.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, record: DocumentRecord, pages: list[Page]) -> None:
        self._pages_path(record.id).write_text(
            json.dumps([{"page": p.number, "text": p.text} for p in pages], ensure_ascii=False),
            encoding="utf-8",
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, sha256, pages, chunks, uploaded_at) "
                "VALUES (:id, :filename, :sha256, :pages, :chunks, :uploaded_at)",
                record.to_dict(),
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return DocumentRecord(**row) if row else None

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return DocumentRecord(**row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at").fetchall()
        return [DocumentRecord(**row) for row in rows]

    def delete(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._pages_path(document_id).unlink(missing_ok=True)

    def read_pages(self, document_id: str) -> list[dict]:
        path = self._pages_path(document_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _pages_path(self, document_id: str) -> Path:
        return self._extracted_dir / f"{document_id}.json"
