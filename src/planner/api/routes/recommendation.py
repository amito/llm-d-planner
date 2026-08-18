"""Recommendation endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from planner.api.dependencies import get_workflow
from planner.orchestration.workflow import RecommendationWorkflow
from planner.shared.schemas.specification import DeploymentSpecification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recommendation"])


class RecommendationRequest(BaseModel):
    """Request for ranked recommendations from a DeploymentSpecification."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "specification": {
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
                    },
                    "enable_estimated": True,
                    "min_quality": 50.0,
                    "max_cost": 10000.0,
                    "include_near_miss": True,
                }
            ]
        }
    }

    specification: DeploymentSpecification
    enable_estimated: bool = True
    min_quality: float | None = None
    max_cost: float | None = None
    include_near_miss: bool = True


@router.post("/generate-recommendations")
def generate_recommendations(
    request: RecommendationRequest,
    workflow: RecommendationWorkflow = Depends(get_workflow),
):
    """
    Generate ranked recommendations from a DeploymentSpecification.

    Accepts a structured specification (output of /generate-specification)
    and returns 4 ranked views of deployment configurations:
    - balanced: Weighted composite score using priorities from spec
    - best_quality: Top configs by model capability
    - lowest_cost: Top configs by price efficiency
    - lowest_latency: Top configs by SLO headroom

    Args:
        request: Request with DeploymentSpecification and optional filters

    Returns:
        RankedRecommendations with 4 ranked lists
    """
    try:
        spec = request.specification

        # Weights from specification priorities
        weights_dict = {
            "quality": spec.priorities.quality.weight,
            "price": spec.priorities.cost.weight,
            "latency": spec.priorities.latency.weight,
        }

        response = workflow.generate_recommendations(
            spec=spec,
            min_quality=request.min_quality,
            max_cost=request.max_cost,
            include_near_miss=request.include_near_miss,
            weights=weights_dict,
            enable_estimated=request.enable_estimated,
        )

        return response

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}",
        ) from e
