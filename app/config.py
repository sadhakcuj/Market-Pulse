from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STOCK_", env_file=".env", extra="ignore")

    api_keys: str = ""  # comma-separated; empty disables auth on /v1/* (dev only)
    model_path: Path = ROOT / "models" / "direction_model.joblib"
    metrics_path: Path = ROOT / "models" / "metrics.json"

    finnhub_api_key: str = Field(default="", validation_alias="FINNHUB_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    sentiment_scorer: str = "auto"  # auto = llm when a key is set, else lexicon; or "lexicon" / "llm"

    news_days: int = Field(default=7, ge=1, le=30)
    max_headlines: int = Field(default=25, ge=1, le=100)
    price_cache_s: int = 600
    news_cache_s: int = 900

    rate_limit_api: str = "120/minute"
    rate_limit_demo: str = "20/minute"
    demo_enabled: bool = True
    cors_origins: str = ""
    log_level: str = "INFO"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
