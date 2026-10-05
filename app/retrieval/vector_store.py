from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb

from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    score: float  # cosine similarity, higher is better

    def to_dict(self) -> dict:
        return asdict(self)


class VectorStore:
    """Thin wrapper around a persistent Chroma collection.

    We compute embeddings ourselves instead of using Chroma's embedding
    function hook, so the embedder stays swappable and testable on its own.
    """

    def __init__(self, path: Path, embedder: Embedder, collection: str = "chunks"):
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def add(self, document_id: str, filename: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            embeddings=self._embedder.embed_documents([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {"document_id": document_id, "filename": filename, "page": c.page, "index": c.index}
                for c in chunks
            ],
        )

    def search(
        self, query: str, top_k: int = 5, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []

        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                filename=meta["filename"],
                page=meta["page"],
                text=text,
                score=round(1.0 - distance, 4),
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()
