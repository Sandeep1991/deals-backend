from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure AI Search
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = "ads"
    azure_search_semantic_config: str = "ads-semantic"
    azure_search_vector_field: str = "contentVector"

    # Search tuning
    search_top_k: int = 10
    search_result_limit: int = 5
    min_search_score: float = 1.0
    min_reranker_score: float = 1.5
    meaning_query_terms: str = "discount,deal,sale,cheap,savings,offer,promo,off"

    # Reply generation: template | ollama | azure_openai
    reply_provider: str = "template"

    # Ollama (local / dev — CPU-friendly small models)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"

    # Azure OpenAI (optional production LLM)
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_openai_api_version: str = "2024-08-01-preview"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
