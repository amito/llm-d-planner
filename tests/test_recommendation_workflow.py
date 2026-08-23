"""Integration test for end-to-end recommendation workflow via API pipeline.

Requires Ollama running for intent extraction.
"""

import json
import logging
from pathlib import Path

import pytest
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"


def _load_scenarios():
    """Load demo scenarios from JSON file."""
    from planner.data._resolver import data_path

    scenarios_path = data_path("configuration/demo_scenarios.json")
    with open(scenarios_path) as f:
        data = json.load(f)
    return data["scenarios"]


@pytest.mark.integration
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_scenario(scenario):
    """Test a single demo scenario through the pipeline."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario['name']}")
    print("=" * 80)
    print(f"\nDescription: {scenario['description']}")
    print(f"\nUser Message: {scenario['user_description']}\n")

    # Stage 1: Extract intent via LLM
    response = requests.post(
        f"{API_BASE}/api/v1/extract-intent",
        json={"text": scenario["user_description"]},
        timeout=60,
    )
    assert response.status_code == 200, f"Intent extraction failed: {response.text}"
    intent = response.json()
    print(f"Extracted intent: use_case={intent['use_case']}, users={intent['user_count']}")

    # Stage 2: Generate specification
    response = requests.post(
        f"{API_BASE}/api/v1/generate-specification",
        json=intent,
        timeout=30,
    )
    assert response.status_code == 200, f"Specification generation failed: {response.text}"
    spec = response.json()

    # Stage 3: Generate recommendations
    response = requests.post(
        f"{API_BASE}/api/v1/generate-recommendations",
        json={"specification": spec, "enable_estimated": False},
        timeout=90,
    )
    assert response.status_code == 200, f"Recommendation generation failed: {response.text}"
    recommendations = response.json()

    assert recommendations["total_configs_evaluated"] > 0, "No configurations evaluated"
    assert len(recommendations["balanced"]) > 0, "No balanced recommendations"

    rec = recommendations["balanced"][0]

    # Display results
    print("\n--- RECOMMENDATION ---")
    print(f"Model: {rec['model_name']}")
    gpu = rec.get("gpu_config", {})
    print(f"GPU Config: {gpu.get('gpu_count', '?')}x {gpu.get('gpu_type', '?')}")
    print(f"  - Tensor Parallel: {gpu.get('tensor_parallel', '?')}")
    print(f"  - Replicas: {gpu.get('replicas', '?')}")

    if rec.get("cost_per_month_usd"):
        print(f"\nCost: ${rec['cost_per_month_usd']:,.0f}/month")

    print("\nPredicted Performance:")
    slo = spec["slo_targets"]
    print(f"  - TTFT p95: {rec.get('predicted_ttft_p95_ms')}ms (target: {slo['ttft_target_ms']}ms)")
    print(f"  - ITL p95: {rec.get('predicted_itl_p95_ms')}ms (target: {slo['itl_target_ms']}ms)")
    print(f"  - E2E p95: {rec.get('predicted_e2e_p95_ms')}ms (target: {slo['e2e_target_ms']}ms)")

    print(f"\nMeets SLO: {'YES' if rec.get('meets_slo') else 'NO'}")

    if rec.get("scores"):
        scores = rec["scores"]
        print(
            f"Scores: Quality={scores['quality_score']:.1f}, "
            f"Price={scores['price_score']}, "
            f"Latency={scores['latency_score']}, "
            f"Balanced={scores['balanced_score']:.1f}"
        )

    # Check against expected recommendation if provided
    if "expected_recommendation" in scenario:
        expected = scenario["expected_recommendation"]
        print("\n--- COMPARISON WITH EXPECTED ---")
        print(
            f"Model Match: {'yes' if rec.get('model_id') == expected['model_id'] else 'no'} "
            f"(expected: {expected['model_id']})"
        )
        print(
            f"GPU Type Match: {'yes' if gpu.get('gpu_type') == expected['gpu_config']['gpu_type'] else 'no'} "
            f"(expected: {expected['gpu_config']['gpu_type']})"
        )
