"""Recommendation-related schemas for GPU configs, scores, and ranked responses."""

from typing import Literal

from pydantic import BaseModel, Field

from .intent import DeploymentIntent
from .specification import DeploymentSpecification, SLOTargets, TrafficProfile


class GPUConfig(BaseModel):
    """GPU configuration specification."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of independent replicas")


class ConfigurationScores(BaseModel):
    """Scores for a deployment configuration (0-100 scale)."""

    quality_score: float = Field(..., description="Model quality/capability score (0-100)")
    price_score: int = Field(..., description="Cost efficiency score - inverse of cost (0-100)")
    latency_score: int = Field(..., description="SLO headroom score (0-100)")
    balanced_score: float = Field(..., description="Weighted composite score (0-100)")
    slo_status: Literal["compliant", "near_miss", "exceeds"] = Field(
        ..., description="SLO compliance status"
    )


class DeploymentConfiguration(BaseModel):
    """Deployment parameters needed to generate YAML files.

    This is the input to generate-deployment. Embedded in each
    DeploymentRecommendation for easy pipeline composition.
    """

    model_id: str = Field(..., description="Model identifier (HuggingFace format)")
    model_name: str | None = Field(None, description="Human-readable model name")
    model_uri: str | None = Field(None, description="Model artifact URI")
    gpu_config: GPUConfig = Field(..., description="GPU type, count, tensor parallelism, replicas")
    use_case: str = Field(..., description="Use case (determines max_model_len)")
    expected_qps: float = Field(..., description="Expected queries per second")
    prompt_tokens: int = Field(..., description="Mean input token length per request")
    output_tokens: int = Field(..., description="Mean output token length per request")
    e2e_target_ms: int = Field(..., description="End-to-end latency target (ms)")


class DeploymentRecommendation(BaseModel):
    """Complete deployment recommendation with all specifications."""

    model_config = {"protected_namespaces": ()}

    # Input intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Recommended configuration (None when no viable config found)
    model_id: str | None = Field(None, description="Recommended model identifier")
    model_name: str | None = Field(None, description="Human-readable model name")
    model_uri: str | None = Field(None, description="Model artifact URI (e.g., OCI registry URI)")
    gpu_config: GPUConfig | None = None

    # Performance predictions (None when no viable config found)
    predicted_ttft_p95_ms: int | None = None
    predicted_itl_p95_ms: int | None = None
    predicted_e2e_p95_ms: int | None = None
    predicted_throughput_qps: float | None = None

    # All percentile metrics from benchmark (for UI to display based on user selection)
    benchmark_metrics: dict | None = Field(
        default=None, description="All percentile metrics from benchmark"
    )

    # Cost estimation (None when no viable config found)
    cost_per_hour_usd: float | None = None
    cost_per_month_usd: float | None = None

    # Metadata
    meets_slo: bool = Field(False, description="Whether configuration meets SLO targets")
    reasoning: str = Field(..., description="Explanation of recommendation choice or error message")
    alternative_options: list[dict] | None = Field(
        default=None, description="Alternative configurations with trade-offs"
    )

    # Multi-criteria scores (added for Solution Ranking feature)
    scores: ConfigurationScores | None = Field(
        default=None, description="Multi-criteria scores for ranking"
    )

    # Deployment configuration (for YAML generation)
    configuration: DeploymentConfiguration | None = Field(
        default=None, description="Deployment parameters for YAML generation"
    )

    def to_alternative_dict(self) -> dict:
        """
        Convert recommendation to alternative option format.

        This is used when building the alternative_options list to avoid
        code duplication across config_finder.py and workflow.py.

        Returns:
            Dictionary with all fields needed for alternative comparison
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_uri": self.model_uri,
            "gpu_config": self.gpu_config.model_dump() if self.gpu_config else None,
            "predicted_ttft_p95_ms": self.predicted_ttft_p95_ms,
            "predicted_itl_p95_ms": self.predicted_itl_p95_ms,
            "predicted_e2e_p95_ms": self.predicted_e2e_p95_ms,
            "predicted_throughput_qps": self.predicted_throughput_qps,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
            "reasoning": self.reasoning,
            "scores": self.scores.model_dump() if self.scores else None,
            "benchmark_metrics": self.benchmark_metrics,
        }


class DeploymentBundle(BaseModel):
    """Generated deployment files — output of generate-deployment."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deployment_id": "chatbot-1000-users-abc123",
                    "namespace": "default",
                    "stack": "vllm",
                    "configuration": {
                        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                        "model_name": "Llama 3.1 8B Instruct",
                        "model_uri": None,
                        "gpu_config": {
                            "gpu_type": "H100",
                            "gpu_count": 1,
                            "tensor_parallel": 1,
                            "replicas": 1,
                        },
                        "use_case": "chatbot_conversational",
                        "expected_qps": 0.87,
                        "prompt_tokens": 512,
                        "output_tokens": 256,
                        "e2e_target_ms": 6280,
                    },
                    "files": {
                        "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\n...",
                        "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n...",
                    },
                }
            ]
        }
    }

    deployment_id: str = Field(..., description="Unique deployment identifier")
    namespace: str = Field(..., description="Kubernetes namespace")
    stack: str = Field(..., description="Deployment stack (vllm or llm-d)")
    configuration: DeploymentConfiguration = Field(
        ..., description="The configuration used to generate these files"
    )
    files: dict[str, str] = Field(
        ...,
        description="Map of filename to YAML content. Files are applied in iteration order "
        "by deploy-bundle-to-cluster, so order matters if resources have dependencies.",
    )


class RankedRecommendations(BaseModel):
    """Response containing multiple ranked recommendation lists.

    Provides 4 different views of the same configurations, each sorted
    by a different criterion to help users explore trade-offs.
    """

    # Filters applied
    min_quality_threshold: float | None = Field(
        default=None, description="Minimum quality score filter applied"
    )
    max_cost_ceiling: float | None = Field(
        default=None, description="Maximum monthly cost filter applied (USD)"
    )
    include_near_miss: bool = Field(
        default=True, description="Whether near-SLO configurations are included"
    )

    # Original specification
    specification: DeploymentSpecification | None = Field(
        default=None, description="The generated deployment specification"
    )

    # Ranked lists (top 5 each, sorted by respective criterion)
    best_quality: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by quality score"
    )
    lowest_cost: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by price score"
    )
    lowest_latency: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by latency score"
    )
    balanced: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by weighted composite score"
    )

    # Statistics
    total_configs_evaluated: int = Field(
        default=0, description="Total number of configurations evaluated"
    )
    configs_after_filters: int = Field(
        default=0, description="Number of configurations after applying filters"
    )

    # Warnings from estimated performance flow
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from estimated performance (e.g., unsupported model architectures)",
    )
