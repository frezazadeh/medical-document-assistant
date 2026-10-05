"""Wires the pieces together in one place.

The API, the MCP server and the eval script each call build_container() and
get the same object graph. Tests pass in a fake embedder and a fake LLM.
"""

from dataclasses import dataclass

from app.agent.graph import build_qa_graph
from app.config import Settings
from app.context import ContextBuilder
from app.ingestion.service import IngestionService
from app.llm import LLMClient, build_llm
from app.observability.tracing import TraceStore
from app.prompts import PromptRegistry
from app.qa import QAService
from app.retrieval.embeddings import Embedder, FastEmbedEmbedder
from app.retrieval.retriever import CrossEncoderReranker, Reranker, Retriever
from app.retrieval.vector_store import VectorStore
from app.storage import DocumentRegistry


@dataclass(frozen=True)
class Container:
    settings: Settings
    registry: DocumentRegistry
    store: VectorStore
    retriever: Retriever
    ingestion: IngestionService
    prompts: PromptRegistry
    context_builder: ContextBuilder
    llm: LLMClient
    traces: TraceStore
    qa: QAService


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    reranker: Reranker | None = None,
) -> Container:
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
    llm = llm or build_llm(settings)

    registry = DocumentRegistry(settings.registry_path, settings.extracted_dir)
    store = VectorStore(settings.chroma_dir, embedder)
    if settings.retrieval_mode == "rerank" and reranker is None:
        reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = Retriever(
        store,
        mode=settings.retrieval_mode,
        reranker=reranker,
        candidates=settings.rerank_candidates,
    )
    ingestion = IngestionService(
        registry=registry,
        store=store,
        uploads_dir=settings.uploads_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    prompts = PromptRegistry()
    context_builder = ContextBuilder(
        prompts,
        max_context_chars=settings.max_context_chars,
        max_history_turns=settings.max_history_turns,
    )
    traces = TraceStore(settings.traces_path)
    graph = build_qa_graph(
        llm=llm,
        retriever=retriever,
        registry=registry,
        prompts=prompts,
        context_builder=context_builder,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
    )
    return Container(
        settings=settings,
        registry=registry,
        store=store,
        retriever=retriever,
        ingestion=ingestion,
        prompts=prompts,
        context_builder=context_builder,
        llm=llm,
        traces=traces,
        qa=QAService(graph, llm, traces),
    )
