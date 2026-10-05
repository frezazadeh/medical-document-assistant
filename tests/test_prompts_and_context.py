import pytest
from jinja2 import UndefinedError

from app.context import NOT_FOUND_ANSWER, ContextBuilder
from app.prompts import PromptRegistry
from app.retrieval.vector_store import RetrievedChunk
from app.storage import DocumentRecord

DOC = DocumentRecord("d1", "report.pdf", "hash", pages=6, chunks=3, uploaded_at="2026-01-01")


def chunk(index: int, page: int, score: float, text: str = "some passage") -> RetrievedChunk:
    return RetrievedChunk(f"d1:{index}", "d1", "report.pdf", page, text, score)


@pytest.fixture
def builder() -> ContextBuilder:
    return ContextBuilder(PromptRegistry(), max_context_chars=100, max_history_turns=2)


def test_registry_loads_all_templates_with_versions():
    registry = PromptRegistry()

    assert registry.names() == ["condense_question", "qa_system", "qa_user"]
    assert registry.get("qa_system").tag.startswith("qa_system@")


def test_missing_template_variable_fails_loudly():
    with pytest.raises(UndefinedError):
        PromptRegistry().get("qa_user").render(question="only the question")


def test_system_prompt_is_static_and_states_the_refusal_sentence(builder):
    assert NOT_FOUND_ANSWER in builder.system_prompt()
    assert builder.system_prompt() == builder.system_prompt()


def test_sources_are_numbered_in_reading_order(builder):
    context = builder.build(
        "What was the dose?",
        retrieved=[chunk(2, page=5, score=0.9), chunk(0, page=1, score=0.7)],
        documents=[DOC],
    )

    assert [(s.ref, s.page) for s in context.sources] == [(1, 1), (2, 5)]
    user_turn = context.messages[-1]["content"]
    assert user_turn.index("[1]\n") < user_turn.index("[2]\n")
    # Only the id is shown. A page number next to it gets copied into the citation.
    assert "page 5" not in user_turn
    assert user_turn.rstrip().endswith("for example [1].")


def test_lowest_scoring_chunks_are_dropped_when_over_budget(builder):
    retrieved = [
        chunk(0, page=1, score=0.9, text="a" * 60),
        chunk(1, page=2, score=0.5, text="b" * 60),
        chunk(2, page=3, score=0.8, text="c" * 30),
    ]

    context = builder.build("q?", retrieved=retrieved, documents=[DOC])

    assert [s.page for s in context.sources] == [1, 3]
    assert context.dropped_chunks == 1


def test_history_is_trimmed_and_starts_with_a_user_turn(builder):
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "assistant", "content": "a2b"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]

    context = builder.build("q4?", retrieved=[chunk(0, 1, 0.9)], documents=[DOC], history=history)

    roles = [m["role"] for m in context.messages]
    assert roles[0] == "user"
    assert len(context.messages) <= 5  # two turns of history plus the new question
    assert context.messages[-1]["content"].count("Question: q4?") == 1
