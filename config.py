from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Project paths
    data_path: Path = BASE_DIR / "data" / "raw_docs"
    index_path: Path = BASE_DIR / "index"
    output_path: Path = BASE_DIR / "output"

    # Retrieval
    chunk_size: int = 1400
    chunk_overlap: int = 250
    top_k: int = 5

    # Workflow
    max_iterations: int = 5

    # Topic defaults
    topic: str = "EV market in Europe 2025"
    scope: str = "Passenger EV adoption, charging infrastructure, regulation, competition"
    focus_areas: list[str] = [
        "market growth",
        "charging infrastructure",
        "policy support",
        "competitive landscape",
        "risks and outlook",
    ]

    # Optional Langfuse, later
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    langfuse_default_user_id: str = "denys"
    langfuse_environment: str = "development"

    # Timeouts
    request_timeout_seconds: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()