from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # treat `QDRANT_API_KEY=` as unset
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str

    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = "EMPTY"  # vLLM accepts any key unless started with --api-key
    embedding_model: str

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
