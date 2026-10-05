"""The question-answering flow as a LangGraph state graph.

    rewrite_query -> retrieve -+-> generate -> check_citations -> END
                               |
                               +-> no_context ------------------> END

Today this is a fixed pipeline with one branch, which a plain function could
also do. It is a graph because the next steps for this project are graph
shaped: a retry edge when citations don't check out, a router in front of
several tools, a human-approval node. Those become new nodes and edges here
without touching retrieval, prompts or the API.

Nodes are plain functions over a typed state. LangGraph is only used for
orchestration; the LLM and the retriever sit behind our own interfaces.
"""

import functools
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.citations import check_citations
from app.context import NOT_FOUND_ANSWER, ContextBuilder, Source
from app.llm import LLMClient
from app.prompts import PromptRegistry
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import RetrievedChunk
from app.storage import DocumentRegistry


class QAState(TypedDict, total=False):
    # input
    question: str
    history: list[dict]
    document_ids: list[str] | None
    # filled in along the way
    search_query: str
    retrieved: list[RetrievedChunk]
    sources: list[Source]
    prompt_versions: list[str]
    model_answer: str  # as generated, before the citation check touches it
    answer: str
    citations: list[Source]
    unknown_refs: list[int]
    citation_repairs: list[dict]
    unsupported: list[str]
    grounded: bool
    usage: dict
    # every node appends one entry; the reducer concatenates them
    steps: Annotated[list[dict], operator.add]


def _step(name: str):
    """Time a node and record what it did, so a trace shows the whole run."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: QAState) -> dict:
            started = time.perf_counter()
            update, info = fn(state)
            ms = round((time.perf_counter() - started) * 1000)
            return {**update, "steps": [{"step": name, "ms": ms, **info}]}

        return wrapper

    return decorator


def build_qa_graph(
    *,
    llm: LLMClient,
    retriever: Retriever,
    registry: DocumentRegistry,
    prompts: PromptRegistry,
    context_builder: ContextBuilder,
    top_k: int,
    min_relevance: float,
):
    condense = prompts.get("condense_question")

    @_step("rewrite_query")
    def rewrite_query(state: QAState):
        history = state.get("history") or []
        if not history:
            return {"search_query": state["question"]}, {"rewritten": False}

        # "And in the placebo group?" retrieves nothing useful on its own.
        prompt = condense.render(history=history[-6:], question=state["question"])
        response = llm.complete(
            system="You rewrite questions for a document search engine.",
            messages=[{"role": "user", "content": prompt}],
        )
        query = response.text.strip().strip('"') or state["question"]
        return {"search_query": query}, {"rewritten": True, "query": query, "prompt": condense.tag}

    @_step("retrieve")
    def retrieve(state: QAState):
        hits = retriever.search(
            state["search_query"], top_k=top_k, document_ids=state.get("document_ids")
        )
        # The gate is on the best hit, not on each chunk: it answers "is this
        # question about these documents at all?". Cosine scores from a small
        # embedding model are too compressed to rank answerable against
        # unanswerable questions on the same topic; the prompt handles those.
        best = max((h.score for h in hits), default=0.0)
        relevant = hits if best >= min_relevance else []
        info = {
            "mode": retriever.mode,
            "hits": [{"chunk_id": h.chunk_id, "page": h.page, "score": h.score} for h in hits],
            "best_score": best,
            "min_relevance": min_relevance,
            "kept": len(relevant),
        }
        return {"retrieved": relevant}, info

    def route_after_retrieve(state: QAState) -> str:
        return "generate" if state["retrieved"] else "no_context"

    @_step("no_context")
    def no_context(state: QAState):
        # Nothing relevant was found, so don't call the model at all. It can't
        # hallucinate an answer it is never asked for, and it costs nothing.
        update = {
            "model_answer": NOT_FOUND_ANSWER,
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "citations": [],
            "unknown_refs": [],
            "citation_repairs": [],
            "unsupported": [],
            "grounded": True,
            "prompt_versions": [],
            "usage": {},
        }
        return update, {}

    @_step("generate")
    def generate(state: QAState):
        scope = state.get("document_ids")
        documents = [d for d in registry.list() if not scope or d.id in scope]
        context = context_builder.build(
            question=state["question"],
            retrieved=state["retrieved"],
            documents=documents,
            history=state.get("history"),
        )
        response = llm.complete(system=context.system, messages=context.messages)
        usage = {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        update = {
            "model_answer": response.text,
            "answer": response.text,
            "sources": context.sources,
            "prompt_versions": context.prompt_versions,
            "usage": usage,
        }
        info = {
            "model": response.model,
            "prompts": context.prompt_versions,
            "sources": len(context.sources),
            "dropped_chunks": context.dropped_chunks,
            **context.stats,
            **usage,
        }
        return update, info

    @_step("check_citations")
    def verify(state: QAState):
        check = check_citations(state["answer"], state["sources"])
        update = {
            "answer": check.answer,  # markers may have been moved to the supporting source
            "citations": check.cited,
            "unknown_refs": check.unknown_refs,
            "citation_repairs": check.repairs,
            "unsupported": check.unsupported,
            "grounded": check.grounded,
        }
        info = {
            "cited": [s.ref for s in check.cited],
            "unknown": check.unknown_refs,
            "repaired": [{"from": r["from"], "to": r["to"]} for r in check.repairs],
            "unsupported": len(check.unsupported),
            "grounded": check.grounded,
        }
        return update, info

    graph = StateGraph(QAState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("no_context", no_context)
    graph.add_node("generate", generate)
    graph.add_node("check_citations", verify)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "no_context"])
    graph.add_edge("generate", "check_citations")
    graph.add_edge("check_citations", END)
    graph.add_edge("no_context", END)
    return graph.compile()
