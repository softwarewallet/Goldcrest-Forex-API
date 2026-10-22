from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Goldcrest Forex API"
    app_env: str = "development"
    log_level: str = "INFO"
    goldcrest_bind_host: str = "127.0.0.1"
    goldcrest_port: int = Field(default=8090, ge=1, le=65535)
    database_path: str = "./data/goldcrest_ai.sqlite3"
    api_key: SecretStr = SecretStr("")

    llama_base_url: str = "http://192.168.29.80:8080"
    llama_model_id: str = ""
    llama_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    llama_max_tokens: int = Field(default=300, ge=32, le=2048)
    max_candle_count: int = Field(default=500, ge=20, le=5000)
    max_snapshot_age_seconds: int = Field(default=120, ge=1, le=3600)
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173,http://192.168.29.80:5173,http://localhost:4173,http://127.0.0.1:4173,http://192.168.29.80:4173", validation_alias=AliasChoices("GOLDCREST_CORS_ORIGINS", "CORS_ORIGINS"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @field_validator("llama_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def database_file(self) -> Path:
        return Path(self.database_path)

    @property
    def api_key_min_length(self) -> int:
        """Short keys are permitted only in development; production always requires 32+ chars."""
        return 8 if self.app_env.strip().lower() in {"development", "dev", "test"} else 32

    @property
    def api_key_configured(self) -> bool:
        value = self.api_key.get_secret_value()
        if value in {
            "",
            "REPLACE_WITH_A_LONG_RANDOM_SECRET",
            "PASTE_A_REAL_GENERATED_SECRET_HERE",
        }:
            return False
        return len(value) >= self.api_key_min_length


@lru_cache
def get_settings() -> Settings:
    return Settings()
