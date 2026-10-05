"""Retrieval strategy on top of the vector store.

"dense"   plain embedding search.
"rerank"  fetch a wider set of dense candidates, then let a cross-encoder
          re-order them and keep the best top_k.

Why the second stage exists: in a clinical report most passages are about the
same drug and the same study, so their embeddings sit very close together.
On the eval set, the passage with the placebo-adjusted blood pressure result
was ranked 9th of 13 by embeddings alone (it says "SBP" where the question says
"systolic blood pressure"), so it never reached the model. A cross-encoder
reads question and passage together and moved it to rank 4.

I tried BM25 with reciprocal rank fusion first because it is cheaper. It moved
that passage from 9th to 6th, still outside the top 5, and pushed another
question's passage down, so I dropped it.
"""

from typing import Literal, Protocol

from app.retrieval.vector_store import RetrievedChunk, VectorStore


class Reranker(Protocol):
    def scores(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return list(self._model.rerank(query, passages))


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        mode: Literal["dense", "rerank"] = "dense",
        reranker: Reranker | None = None,
        candidates: int = 20,
    ):
        if mode == "rerank" and reranker is None:
            raise ValueError("mode 'rerank' needs a reranker")
        self._store = store
        self._reranker = reranker
        self._candidates = candidates
        self.mode = mode

    def search(
        self, query: str, top_k: int = 5, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        if self.mode == "dense":
            return self._store.search(query, top_k, document_ids)

        candidates = self._store.search(query, max(self._candidates, top_k), document_ids)
        if len(candidates) <= 1:
            return candidates

        scores = self._reranker.scores(query, [c.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        # Each chunk keeps its cosine similarity as `score` (the relevance gate
        # and the UI use it); the reranker only decides the order.
        return [candidates[i] for i in order[:top_k]]
