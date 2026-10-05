"""Test doubles so the suite runs offline: no model download, no LLM server."""

import hashlib
import math
import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.llm import LLMResponse
from app.main import create_app

API_KEY = "test-key"

STUDY_TEXT = (
    "Study ABC-1 enrolled adults with asthma. A total of 240 participants were randomized. "
    "The primary endpoint was the change in FEV1 at week 8.\f"
    "Safety. Headache was reported in 12 participants. No deaths occurred during the study."
)


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words are similar, others are not."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages})
        text = self.replies.pop(0) if self.replies else "The answer is in the document [1]."
        return LLMResponse(text=text, model=self.model, input_tokens=10, output_tokens=5)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        api_key=API_KEY,
        retrieval_mode="dense",
        min_relevance=0.2,
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def container(settings, llm):
    return build_container(settings, embedder=FakeEmbedder(), llm=llm)


@pytest.fixture
def client(settings, container):
    with TestClient(create_app(settings, container)) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict:
    return {"X-API-Key": API_KEY}
