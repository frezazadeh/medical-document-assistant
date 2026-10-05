import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from app.context import Source
from app.llm import LLMClient
from app.observability.tracing import TraceStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    trace_id: str
    answer: str
    citations: list[Source]
    grounded: bool
    unsupported: list[str]
    model: str
    latency_ms: int


class QAService:
    """Runs the graph for one question and writes the trace. The API, the MCP
    server and the eval script all go through here, so they behave the same."""

    def __init__(self, graph, llm: LLMClient, traces: TraceStore):
        self._graph = graph
        self._llm = llm
        self._traces = traces

    def ask(
        self,
        question: str,
        document_ids: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> Answer:
        trace_id = uuid.uuid4().hex[:16]
        started = time.perf_counter()

        state = self._graph.invoke(
            {
                "question": question.strip(),
                "history": history or [],
                "document_ids": document_ids or None,
                "steps": [],
            }
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        self._traces.append(
            {
                "trace_id": trace_id,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": question,
                "search_query": state["search_query"],
                "document_ids": document_ids,
                "provider": self._llm.provider,
                "model": self._llm.model,
                "prompt_versions": state["prompt_versions"],
                "steps": state["steps"],
                "sources": [asdict(s) for s in state["sources"]],
                "model_answer": state["model_answer"],
                "answer": state["answer"],
                "cited_refs": [s.ref for s in state["citations"]],
                "unknown_refs": state["unknown_refs"],
                "citation_repairs": state["citation_repairs"],
                "unsupported": state["unsupported"],
                "grounded": state["grounded"],
                "usage": state["usage"],
                "latency_ms": latency_ms,
            }
        )
        logger.info(
            "question answered",
            extra={
                "trace_id": trace_id,
                "grounded": state["grounded"],
                "citations": len(state["citations"]),
                "latency_ms": latency_ms,
            },
        )
        if not state["grounded"]:
            logger.warning(
                "answer is not grounded",
                extra={
                    "trace_id": trace_id,
                    "unknown_refs": state["unknown_refs"],
                    "unsupported": len(state["unsupported"]),
                },
            )

        return Answer(
            trace_id=trace_id,
            answer=state["answer"],
            citations=state["citations"],
            grounded=state["grounded"],
            unsupported=state["unsupported"],
            model=self._llm.model,
            latency_ms=latency_ms,
        )
