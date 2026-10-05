from app.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse


def build_llm(settings: Settings) -> LLMClient:
    # Imports are local so that running with Ollama doesn't require the
    # Anthropic SDK to be importable, and the other way round.
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            api_key=settings.anthropic_api_key,
        )

    from app.llm.ollama_client import OllamaClient

    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_ctx=settings.ollama_num_ctx,
        timeout=settings.llm_timeout_s,
    )


__all__ = ["LLMClient", "LLMError", "LLMResponse", "build_llm"]
