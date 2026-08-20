"""Configuration for the Planner library."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PlannerConfig(BaseModel):
    """Configuration for the Planner library.

    All fields are optional — Planner() with no config uses bundled data
    and works without any external services.

    Example:
        config = PlannerConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            quality_auto_update=True,
            aa_api_key="aa-...",
        )
        p = Planner(config)
    """

    # Data directory override (default: bundled package data)
    data_dir: Path | None = Field(
        default=None,
        description="Custom data directory for config/quality files. "
        "If None, uses data bundled in the wheel.",
    )

    # Intent extraction (optional — only needed for extract_intent())
    llm_provider: str | None = Field(
        default=None,
        description='LLM provider: "ollama", "openai", or "vertex"',
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider (OpenAI, Vertex)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoints",
    )
    llm_model: str | None = Field(
        default=None,
        description="Model name override for the LLM provider",
    )

    # Quality scoring data
    quality_auto_update: bool = Field(
        default=False,
        description="Fetch fresh Arena/AA data on init when cached data is stale",
    )
    quality_cache_dir: Path | None = Field(
        default=None,
        description="Directory for runtime quality data cache. "
        "Default: .quality_cache/ in current directory.",
    )
    aa_api_key: str | None = Field(
        default=None,
        description="Artificial Analysis API key (required for AA data refresh)",
    )

    # HuggingFace (for estimated performance via capacity planner)
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace API token for model config lookups",
    )

    # Model Catalog sync
    model_catalog_url: str | None = Field(
        default=None,
        description="Model Catalog API URL for sync_model_catalog()",
    )
    model_catalog_token: str | None = Field(
        default=None,
        description="Auth token for Model Catalog API",
    )
