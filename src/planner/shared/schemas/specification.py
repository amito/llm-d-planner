"""Specification-related schemas for the deployment pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

from .intent import DeploymentIntent


class TrafficProfile(BaseModel):
    """GuideLLM traffic profile for the deployment."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens")
    output_tokens: int = Field(..., description="Target output length in tokens")
    expected_qps: float | None = Field(None, description="Expected queries per second")


class SLORange(BaseModel):
    """Recommended min/max range for an SLO target. Informational; not enforced."""

    min: int
    max: int


class SLOTargets(BaseModel):
    """Service Level Objective targets for the deployment."""

    ttft_target_ms: int = Field(..., description="Time to First Token target (ms)")
    itl_target_ms: int = Field(..., description="Inter-Token Latency target (ms/token)")
    e2e_target_ms: int = Field(..., description="End-to-end latency target (ms)")
    percentile: Literal["p90", "p95", "p99"] = Field(
        default="p95", description="Percentile for SLO comparison"
    )
    ttft_range: SLORange | None = Field(default=None, description="Recommended TTFT range")
    itl_range: SLORange | None = Field(default=None, description="Recommended ITL range")
    e2e_range: SLORange | None = Field(default=None, description="Recommended E2E range")


class WorkloadProfile(BaseModel):
    """Workload characteristics derived from use case and user count."""

    prompt_tokens: int = Field(..., description="Mean input token length per request")
    output_tokens: int = Field(..., description="Mean output token length per request")
    expected_qps: float = Field(
        ...,
        description="Expected queries per second the deployment must support (includes peak capacity buffer)",
    )


class QualityWeights(BaseModel):
    """Per-use-case category weights for quality scoring."""

    categories: dict[str, int] = Field(..., description="Category name to relative weight mapping")


class PriorityEntry(BaseModel):
    """A single priority dimension with level and resolved weight."""

    priority: Literal["low", "medium", "high"]
    weight: int


class Priorities(BaseModel):
    """Quality/cost/latency priority levels with resolved weights."""

    quality: PriorityEntry
    cost: PriorityEntry
    latency: PriorityEntry


class DeploymentSpecification(BaseModel):
    """Complete deployment specification — output of generate-specification."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "intent": {
                        "use_case": "chatbot_conversational",
                        "user_count": 1000,
                        "domain_specialization": ["general"],
                        "preferred_gpu_types": [],
                        "preferred_models": [],
                        "quality_priority": "medium",
                        "cost_priority": "medium",
                        "latency_priority": "high",
                    },
                    "slo_targets": {
                        "ttft_target_ms": 200,
                        "itl_target_ms": 24,
                        "e2e_target_ms": 6280,
                        "percentile": "p95",
                        "ttft_range": {"min": 100, "max": 500},
                        "itl_range": {"min": 15, "max": 50},
                        "e2e_range": {"min": 3940, "max": 13300},
                    },
                    "workload_profile": {
                        "prompt_tokens": 512,
                        "output_tokens": 256,
                        "expected_qps": 0.87,
                    },
                    "quality_weights": {
                        "categories": {
                            "overall": 4,
                            "instruction_following": 3,
                            "multi_turn": 3,
                            "creative_writing": 2,
                            "hard_prompts": 2,
                        }
                    },
                    "priorities": {
                        "quality": {"priority": "medium", "weight": 4},
                        "cost": {"priority": "medium", "weight": 4},
                        "latency": {"priority": "high", "weight": 2},
                    },
                }
            ]
        }
    }

    intent: DeploymentIntent
    slo_targets: SLOTargets
    workload_profile: WorkloadProfile
    quality_weights: QualityWeights | None = Field(
        default=None,
        description="Read-only. Shows the per-use-case category weights used for quality scoring. Changes are not currently honored.",
    )
    priorities: Priorities
