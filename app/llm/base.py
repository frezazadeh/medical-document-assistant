from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """The model backend failed or declined. Carries a message safe to show to a client."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        """messages: [{"role": "user" | "assistant", "content": str}, ...]"""
        ...
