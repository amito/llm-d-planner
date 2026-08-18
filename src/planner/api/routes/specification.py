"""Specification-related endpoints (SLO defaults, workload profiles)."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from planner.api.dependencies import get_traffic_generator
from planner.data._resolver import data_path as resolve_data_path
from planner.shared.schemas.intent import DeploymentIntent
from planner.shared.schemas.specification import DeploymentSpecification
from planner.specification.service import SpecificationService
from planner.specification.traffic_profile import TrafficProfileGenerator, _round_to_nearest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["specification"])


def _calculate_percentile_value(min_val: int, max_val: int, percentile: float = 0.75) -> int:
    """Calculate value at given percentile between min and max, rounded to nearest 5."""
    value = min_val + (max_val - min_val) * percentile
    return _round_to_nearest(value, 5)


def _load_use_case_config(use_case: str) -> dict:
    """Load and validate use case config from SLO workload file.

    Raises HTTPException 404 if file or use case not found.
    """
    json_path = resolve_data_path("configuration/usecase_slo_workload.json")
    if not json_path.exists():
        logger.error("SLO workload config not found at: %s", json_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="SLO workload configuration not found"
        )

    with open(json_path) as f:
        data: dict = json.load(f)

    use_case_data: dict = data.get("use_case_slo_workload", {}).get(use_case)
    if not use_case_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Use case '{use_case}' not found"
        )
    return use_case_data


@router.get("/slo-defaults/{use_case}")
async def get_slo_defaults(use_case: str):
    """Get default SLO values for a use case.

    Returns SLO targets at the 75th percentile between min and max,
    rounded to the nearest 5.
    """
    try:
        use_case_data = _load_use_case_config(use_case)
        slo_targets = use_case_data.get("slo_targets", {})

        ttft = slo_targets["ttft_ms"]
        itl = slo_targets["itl_ms"]
        e2e = slo_targets["e2e_ms"]

        defaults = {
            "use_case": use_case,
            "description": use_case_data.get("description", ""),
            "ttft_ms": {
                "min": ttft["min"],
                "max": ttft["max"],
                "default": _calculate_percentile_value(ttft["min"], ttft["max"], 0.75),
            },
            "itl_ms": {
                "min": itl["min"],
                "max": itl["max"],
                "default": _calculate_percentile_value(itl["min"], itl["max"], 0.75),
            },
            "e2e_ms": {
                "min": e2e["min"],
                "max": e2e["max"],
                "default": _calculate_percentile_value(e2e["min"], e2e["max"], 0.75),
            },
        }

        return {"success": True, "slo_defaults": defaults}

    except HTTPException:
        raise
    except KeyError as e:
        logger.error("Missing SLO data for %s: %s", use_case, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing SLO data: {e}"
        ) from e
    except Exception as e:
        logger.error("Failed to get SLO defaults for %s: %s", use_case, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/workload-profile/{use_case}")
async def get_workload_profile(use_case: str):
    """Get workload profile for a use case.

    Returns the token configuration and peak multiplier used for
    capacity planning and recommendation generation.
    """
    try:
        use_case_data = _load_use_case_config(use_case)
        workload = use_case_data.get("workload", {})

        prompt_tokens = workload["prompt_tokens"]
        output_tokens = workload["output_tokens"]
        active_fraction = workload["active_fraction"]
        requests_per_active_user_per_min = workload["requests_per_active_user_per_min"]

        return {
            "success": True,
            "use_case": use_case,
            "description": use_case_data.get("description", ""),
            "workload_profile": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "active_fraction": active_fraction,
                "requests_per_active_user_per_min": requests_per_active_user_per_min,
            },
        }

    except HTTPException:
        raise
    except KeyError as e:
        logger.error("Missing workload data for %s: %s", use_case, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing workload data: {e}"
        ) from e
    except Exception as e:
        logger.error("Failed to get workload profile for %s: %s", use_case, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/expected-rps/{use_case}")
async def get_expected_rps(use_case: str, user_count: int = 1000):
    """Calculate expected RPS for a use case based on workload patterns.

    Uses research-backed workload distribution parameters:
    - active_fraction: percentage of users active at any time
    - requests_per_active_user_per_min: request rate per active user

    Formula: expected_rps = (user_count * active_fraction * requests_per_min) / 60
    """
    try:
        use_case_data = _load_use_case_config(use_case)
        workload = use_case_data.get("workload", {})

        active_fraction = workload["active_fraction"]
        requests_per_min = workload["requests_per_active_user_per_min"]
        peak_multiplier = workload.get("peak_multiplier", 2.0)

        expected_concurrent = int(user_count * active_fraction)
        expected_rps = (expected_concurrent * requests_per_min) / 60
        expected_rps = max(1, round(expected_rps, 2))

        peak_rps = expected_rps * peak_multiplier

        return {
            "success": True,
            "use_case": use_case,
            "user_count": user_count,
            "workload_params": {
                "active_fraction": active_fraction,
                "requests_per_active_user_per_min": requests_per_min,
                "peak_multiplier": peak_multiplier,
            },
            "expected_rps": expected_rps,
            "expected_concurrent_users": expected_concurrent,
            "peak_rps": round(peak_rps, 2),
        }

    except HTTPException:
        raise
    except KeyError as e:
        logger.error("Missing workload data for %s: %s", use_case, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing workload data: {e}"
        ) from e
    except Exception as e:
        logger.error("Failed to calculate expected RPS for %s: %s", use_case, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/generate-specification")
async def generate_specification(
    intent: DeploymentIntent,
    generator: TrafficProfileGenerator = Depends(get_traffic_generator),
) -> DeploymentSpecification:
    """Generate a complete deployment specification from structured intent.

    No LLM required. Reads from static config files to generate:
    - SLO targets with ranges (based on latency priority)
    - Workload profile (traffic pattern + peak multiplier)
    - Quality weights (category importance for model scoring)
    - Priorities (resolved priority levels to numeric weights)

    Args:
        intent: Deployment intent with use_case, user_count, and optional preferences

    Returns:
        Complete DeploymentSpecification ready for recommendation engine

    Raises:
        HTTPException: 422 if use_case unknown or data missing
        HTTPException: 500 for unexpected errors
    """
    try:
        # Use SpecificationService for single source of truth
        service = SpecificationService(traffic_gen=generator)
        return service.generate(intent)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        logger.error(f"Invalid configuration data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except KeyError as e:
        logger.error(f"Missing configuration data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing configuration data: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Failed to generate specification: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
