import anthropic

from app.llm.base import LLMError, LLMResponse


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str, effort: str = "medium", api_key: str | None = None):
        self.model = model
        self._effort = effort
        # Without an explicit key the SDK resolves credentials from the environment.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                # Current Claude models reject temperature; effort is the knob for
                # how much reasoning goes into an answer.
                output_config={"effort": self._effort},
                # Clinical text can trip the safety classifiers by accident. With this
                # the API retries a declined request on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Unknown Anthropic model '{self.model}'.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached. Try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not connect to the Anthropic API.") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
