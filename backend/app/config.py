"""Application settings, loaded from environment / .env (see .env.example)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/ directory; data files and the SQLite db live under app/data.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (D1: chat and embedding models are distinct; embedding used from Phase 2).
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Configurable so any model works. Some reasoning models reject temperature != 1;
    # set CHAT_TEMPERATURE=1 for those. 0.0 suits gpt-4o-mini / gpt-5-nano.
    chat_temperature: float = 0.0
    request_timeout: float = 30.0  # seconds; bounds LLM calls so they can't hang forever

    # Persistence.
    db_path: str = str(DATA_DIR / "app.db")
    chroma_path: str = str(DATA_DIR / "chroma")  # Chroma persistent store (Phase 2)

    # Auth / JWT (Phase 3). Set a strong JWT_SECRET in .env for anything non-local.
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # CORS (comma-separated origins).
    cors_origins: str = "http://localhost:3000"

    # Phase 1 placeholder identity; replaced by the authenticated user in Phase 3 (D8 seam).
    placeholder_user_id: str = "dev-user"

    # Phase 4: run the lightweight LLM guardrail classifier in addition to the regex
    # pre-filter. Adds one cheap LLM call per request; turn off to save latency/cost.
    enable_guardrail_llm: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
