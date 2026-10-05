import httpx

from app.llm.base import LLMError, LLMResponse


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, num_ctx: int = 4096, timeout: float = 180.0):
        self.model = model
        self._num_ctx = num_ctx
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            # Deterministic output: this is extraction from sources, not creative writing.
            "options": {"temperature": 0, "num_ctx": self._num_ctx},
        }
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                "Cannot reach Ollama. Is it running? Start it with `ollama serve`."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise LLMError(
                    f"Ollama does not have model '{self.model}'. Run `ollama pull {self.model}`."
                ) from exc
            raise LLMError(f"Ollama returned HTTP {exc.response.status_code}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama timed out while generating the answer.") from exc

        body = response.json()
        return LLMResponse(
            text=body["message"]["content"].strip(),
            model=body.get("model", self.model),
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )
