"""Assembles everything the model sees for one question.

There are four inputs and each has one place where it enters:

  static    instructions            -> system prompt (same for every request)
  static    document inventory      -> top of the user turn
  dynamic   retrieved excerpts      -> <sources>, numbered, within a size budget
  dynamic   conversation history    -> earlier turns, last N only

Nothing else in the code base builds prompt strings. If a new kind of context
is needed (patient metadata, a glossary, tool results) it gets added here.
"""

from dataclasses import dataclass, field

from app.prompts import PromptRegistry
from app.retrieval.vector_store import RetrievedChunk
from app.storage import DocumentRecord

NOT_FOUND_ANSWER = "I could not find this in the provided documents."


@dataclass(frozen=True)
class Source:
    """A retrieved chunk as it is shown to the model. `ref` is the id it cites."""

    ref: int
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class ModelContext:
    system: str
    messages: list[dict]
    sources: list[Source]
    prompt_versions: list[str]
    dropped_chunks: int = 0
    stats: dict = field(default_factory=dict)


class ContextBuilder:
    def __init__(self, prompts: PromptRegistry, max_context_chars: int, max_history_turns: int):
        self._system = prompts.get("qa_system")
        self._user = prompts.get("qa_user")
        self._max_context_chars = max_context_chars
        self._max_history_turns = max_history_turns

    def system_prompt(self) -> str:
        return self._system.render(not_found_answer=NOT_FOUND_ANSWER)

    def build(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        documents: list[DocumentRecord],
        history: list[dict] | None = None,
    ) -> ModelContext:
        sources, dropped = self._select_sources(retrieved)
        user_turn = self._user.render(documents=documents, sources=sources, question=question)
        history = self._trim_history(history or [])

        return ModelContext(
            system=self.system_prompt(),
            messages=[*history, {"role": "user", "content": user_turn}],
            sources=sources,
            prompt_versions=[self._system.tag, self._user.tag],
            dropped_chunks=dropped,
            stats={
                "source_chars": sum(len(s.text) for s in sources),
                "history_turns": len(history),
            },
        )

    def _select_sources(self, retrieved: list[RetrievedChunk]) -> tuple[list[Source], int]:
        """Best chunks first until the budget is used up, then numbered in reading order."""
        picked: list[RetrievedChunk] = []
        used = 0
        for chunk in sorted(retrieved, key=lambda c: c.score, reverse=True):
            if picked and used + len(chunk.text) > self._max_context_chars:
                continue
            picked.append(chunk)
            used += len(chunk.text)

        # Reading order (document, then page) is easier for the model to follow
        # than relevance order, and it makes the cited ids look less arbitrary.
        picked.sort(key=lambda c: (c.filename, c.page, c.chunk_id))
        sources = [
            Source(
                ref=i,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page=c.page,
                text=c.text,
                score=c.score,
            )
            for i, c in enumerate(picked, start=1)
        ]
        return sources, len(retrieved) - len(picked)

    def _trim_history(self, history: list[dict]) -> list[dict]:
        turns = [
            {"role": t["role"], "content": t["content"]}
            for t in history
            if t.get("role") in {"user", "assistant"} and t.get("content")
        ]
        turns = turns[-2 * self._max_history_turns :]
        # Model APIs expect the conversation to open with a user turn.
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        return turns
