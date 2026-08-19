"""Intent-related schemas for deployment requirements."""

from typing import Literal

from pydantic import BaseModel, Field


class GpuPreference(BaseModel):
    """GPU type preference with optional count limit."""

    gpu_type: str = Field(..., description="GPU type name (e.g., H100, L4)")
    max_count: int | None = Field(
        default=None, description="Maximum GPU count for this type (None = no limit)"
    )


class DeploymentIntent(BaseModel):
    """Extracted deployment requirements from user conversation."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "use_case": "chatbot_conversational",
                    "user_count": 1000,
                },
                {
                    "use_case": "code_generation_detailed",
                    "user_count": 500,
                    "domain_specialization": ["code"],
                    "preferred_gpu_types": ["L4", {"gpu_type": "H100", "max_count": 4}],
                    "preferred_models": ["meta-llama/Llama-3.1-8B-Instruct"],
                    "quality_priority": "high",
                    "cost_priority": "medium",
                    "latency_priority": "low",
                },
            ]
        }
    }

    use_case: Literal[
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_generation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ] = Field(..., description="Primary use case type")

    user_count: int = Field(..., description="Number of users or scale")

    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"],
        description="Domain requirements (general, code, multilingual, enterprise)",
    )

    # Hardware preference extracted from natural language
    preferred_gpu_types: list[str | GpuPreference] = Field(
        default_factory=list,
        description="List of preferred GPU types. Plain strings or GpuPreference objects with max_count.",
    )

    preferred_models: list[str] = Field(
        default_factory=list,
        description="List of user's preferred model IDs (HuggingFace format, empty = any model). "
        "Can be catalog model_ids or arbitrary HF repo IDs.",
    )

    # Priority hints extracted from natural language (used for weight calculation)
    quality_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Quality importance"
    )
    cost_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Cost sensitivity (high = very cost sensitive)"
    )
    latency_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Latency importance"
    )


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None
