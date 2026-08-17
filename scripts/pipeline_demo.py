#!/usr/bin/env -S uv run python
"""Interactive pipeline demonstration script.

Demonstrates the programmatic API pipeline stages by calling endpoints:
1. generate-specification — DeploymentIntent → DeploymentSpecification
2. generate-recommendations — DeploymentSpecification → RankedRecommendations
3. generate-deployment — DeploymentRecommendation → DeploymentBundle

Requires a running backend (default: http://localhost:8000).
Start with: make start
"""

import argparse
import json
import sys

import requests
from rich.console import Console
from rich.syntax import Syntax

console = Console()

# ============================================================================
# Hardcoded DeploymentIntent — modify this to try different scenarios
# ============================================================================
DEMO_INTENT = {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "quality_priority": "high",
    "cost_priority": "low",
    "latency_priority": "medium",
    "preferred_gpu_types": ["H100", "H200"],
    # "preferred_models": ["meta-llama/llama-3.1-8b-instruct"],
    "domain_specialization": ["general"],
}


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_json(label: str, data: dict):
    print(f"\n{label}:")
    syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai", padding=1)
    console.print(syntax)


def print_recommendation_tables(data: dict):
    """Print top 5 recommendations in each category as tables."""
    categories = [
        ("Balanced", "balanced"),
        ("Best Quality", "best_quality"),
        ("Lowest Cost", "lowest_cost"),
        ("Lowest Latency", "lowest_latency"),
    ]

    print(f"\nTotal configurations evaluated: {data['total_configs_evaluated']}")
    print(f"Configurations after filters: {data['configs_after_filters']}")

    for display_name, key in categories:
        recs = data.get(key, [])[:5]
        if not recs:
            print(f"\n  {display_name}: (none)")
            continue

        print(f"\n  {display_name}:")
        print(f"  {'#':<3} {'Model':<40} {'GPU':<15} {'$/mo':<10} "
              f"{'Qual':<6} {'Price':<6} {'Lat':<6} {'Bal':<6}")
        print(f"  {'-'*3} {'-'*40} {'-'*15} {'-'*10} "
              f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}")

        for i, rec in enumerate(recs, 1):
            model = (rec.get("model_name") or "unknown")[:40]
            gpu_cfg = rec.get("gpu_config") or {}
            gpu = f"{gpu_cfg.get('gpu_count', '?')}x {gpu_cfg.get('gpu_type', '?')}" if gpu_cfg else "N/A"
            cost = rec.get("cost_per_month_usd")
            cost_str = f"${cost:,.0f}" if cost else "N/A"
            scores = rec.get("scores") or {}
            qual = f"{scores['quality_score']:.1f}" if "quality_score" in scores else "-"
            price = f"{scores['price_score']:.0f}" if "price_score" in scores else "-"
            lat = f"{scores['latency_score']:.0f}" if "latency_score" in scores else "-"
            bal = f"{scores['balanced_score']:.1f}" if "balanced_score" in scores else "-"

            print(f"  {i:<3} {model:<40} {gpu:<15} {cost_str:<10} "
                  f"{qual:<6} {price:<6} {lat:<6} {bal:<6}")


def call_api(url: str, payload: dict, timeout: int = 60) -> dict:
    """Call an API endpoint and return the JSON response. Exit on error."""
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text[:500]}")
        sys.exit(1)


def pause(message: str = "Press Enter to continue..."):
    input(f"\n{message}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive pipeline demonstration for programmatic API users"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code != 200:
            print(f"Backend health check failed: {r.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to backend at {base_url}")
        print(f"Make sure the backend is running: make start")
        sys.exit(1)

    # ========================================================================
    # Show the starting point
    # ========================================================================
    print_header("Pipeline Demo — Programmatic API")
    print("\nThis demo walks through the pipeline stages, showing the JSON")
    print("input and output at each step. Modify DEMO_INTENT in the script")
    print("to try different scenarios.")
    print("\nWe start with a DeploymentIntent (skipping LLM intent extraction):")
    print_json("DeploymentIntent", DEMO_INTENT)

    pause("Press Enter to submit this intent to POST /generate-specification...")

    # ========================================================================
    # Stage 1: generate-specification
    # ========================================================================
    print_header("Stage 1: POST /generate-specification")
    print("\nThe input is the DeploymentIntent shown above.")

    specification = call_api(
        f"{base_url}/api/v1/generate-specification", DEMO_INTENT
    )

    print_json("Response — DeploymentSpecification", specification)

    pause("The output above is the input for the next stage, wrapped in a "
          "request object. Press Enter to see the request...")

    # ========================================================================
    # Stage 2: generate-recommendations
    # ========================================================================
    print_header("Stage 2: POST /generate-recommendations")

    recommendation_request = {
        "specification": specification,
        "enable_estimated": True,
        "include_near_miss": True,
    }

    print_json("Request — RecommendationRequest", recommendation_request)

    pause("Press Enter to submit this request...")

    recommendations = call_api(
        f"{base_url}/api/v1/generate-recommendations",
        recommendation_request,
        timeout=90,
    )

    print("\nResponse — RankedRecommendations (summary):")
    print_recommendation_tables(recommendations)

    if not recommendations.get("balanced"):
        print("\nNo recommendations found. Pipeline stopped.")
        sys.exit(0)

    print("\nFor demo purposes, we are choosing the first recommendation "
          "in the balanced category.")

    pause("Press Enter to see the generate-deployment request...")

    # ========================================================================
    # Stage 3: generate-deployment
    # ========================================================================
    print_header("Stage 3: POST /generate-deployment")

    top_recommendation = recommendations["balanced"][0]
    configuration = top_recommendation["configuration"]

    deployment_request = {
        "configuration": configuration,
        "namespace": "default",
        "stack": "vllm",
    }

    print_json("Request — GenerateDeploymentRequest", deployment_request)

    pause("Press Enter to submit this request...")

    deployment_bundle = call_api(
        f"{base_url}/api/v1/generate-deployment", deployment_request
    )

    # Show the bundle without the full YAML content (just file names)
    bundle_summary = {
        "deployment_id": deployment_bundle["deployment_id"],
        "namespace": deployment_bundle["namespace"],
        "stack": deployment_bundle["stack"],
        "files": {k: f"({len(v)} bytes of YAML)" for k, v in deployment_bundle["files"].items()},
    }
    print_json("Response — DeploymentBundle (summary)", bundle_summary)

    # Show actual YAML content
    for filename, content in deployment_bundle["files"].items():
        print(f"\n--- {filename} ---")
        syntax = Syntax(content.strip(), "yaml", theme="monokai", padding=1)
        console.print(syntax)

    # ========================================================================
    # Done
    # ========================================================================
    print_header("Pipeline Complete")
    print("\nThe DeploymentBundle above can be deployed to a cluster with:")
    print("  POST /api/v1/deploy-bundle-to-cluster")
    print(f'  {{"bundle": <the full DeploymentBundle JSON>}}')
    print("\nTo try different scenarios, modify DEMO_INTENT at the top of this script.")


if __name__ == "__main__":
    main()
