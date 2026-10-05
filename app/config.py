from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MDA_", extra="ignore", populate_by_name=True
    )

    api_key: str = "dev-key-change-me"
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    max_upload_mb: int = 20

    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    # Ollama truncates silently past this, so keep max_context_chars well inside it.
    ollama_num_ctx: int = 4096
    llm_timeout_s: float = 180.0
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "medium"
    # No MDA_ prefix so the usual variable name works from .env as well as the shell.
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_mode: Literal["dense", "rerank"] = "rerank"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    top_k: int = 5
    # Cosine similarity below this is treated as "nothing relevant found".
    min_relevance: float = 0.5
    max_context_chars: int = 6000
    max_history_turns: int = 4

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.sqlite"

    @property
    def traces_path(self) -> Path:
        return self.data_dir / "traces.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
