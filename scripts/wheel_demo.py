#!/usr/bin/env python3
"""Interactive demo of the Planner Python library (wheel API).

This script demonstrates the pipeline using the Planner class directly,
without a running server. It mirrors scripts/pipeline_demo.py but uses
the library API instead of REST calls.

Once llm-d-planner is published to PyPI, just:

    pip install llm-d-planner
    python wheel_demo.py

Until then, build and install the wheel locally:

    # 1. Build the wheel from the repo
    cd ~/go/src/github.com/llm-d-planner
    uv build

    # 2. Create an isolated test directory with its own venv
    mkdir /tmp/planner-wheel-test
    cd /tmp/planner-wheel-test
    python3 -m venv .venv
    source .venv/bin/activate

    # 3. Install the wheel (no access to the repo)
    pip install ~/go/src/github.com/llm-d-planner/dist/llm_d_planner-*.whl

    # 4. Copy this script and run it
    cp ~/go/src/github.com/llm-d-planner/scripts/wheel_demo.py .
    python wheel_demo.py
"""

import json
import sys

from planner import (
    DeploymentConfiguration,
    DeploymentIntent,
    GPUConfig,
    Planner,
    PlannerError,
)

# ============================================================================
# Hardcoded DeploymentIntent — modify to try different scenarios
# ============================================================================
DEMO_INTENT = DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
    quality_priority="high",
    cost_priority="low",
    latency_priority="medium",
    preferred_gpu_types=["H100", "H200"],
    # preferred_models=["meta-llama/llama-3.1-8b-instruct"],
    domain_specialization=["general"],
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_json(label: str, data):
    """Print a Pydantic model or dict as formatted JSON."""
    print(f"\n{label}:")
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    print(json.dumps(data, indent=2, default=str))


def print_recommendation_tables(recs):
    """Print top 5 recommendations in each category."""
    categories = [
        ("Balanced", recs.balanced),
        ("Best Quality", recs.best_quality),
        ("Lowest Cost", recs.lowest_cost),
        ("Lowest Latency", recs.lowest_latency),
    ]

    print(f"\nTotal configurations evaluated: {recs.total_configs_evaluated}")
    print(f"Configurations after filters: {recs.configs_after_filters}")

    for display_name, items in categories:
        items = items[:5]
        if not items:
            print(f"\n  {display_name}: (none)")
            continue

        print(f"\n  {display_name}:")
        print(
            f"  {'#':<3} {'Model':<40} {'GPU':<15} {'$/mo':<10} "
            f"{'Qual':<6} {'Price':<6} {'Lat':<6} {'Bal':<6}"
        )
        print(
            f"  {'-'*3} {'-'*40} {'-'*15} {'-'*10} "
            f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}"
        )

        for i, rec in enumerate(items, 1):
            model = (rec.model_name or "unknown")[:40]
            gpu = f"{rec.gpu_config.gpu_count}x {rec.gpu_config.gpu_type}" if rec.gpu_config else "N/A"
            cost = f"${rec.cost_per_month_usd:,.0f}" if rec.cost_per_month_usd else "N/A"
            scores = rec.scores
            qual = f"{scores.quality_score:.1f}" if scores else "-"
            price = f"{scores.price_score:.0f}" if scores else "-"
            lat = f"{scores.latency_score:.0f}" if scores else "-"
            bal = f"{scores.balanced_score:.1f}" if scores else "-"

            print(
                f"  {i:<3} {model:<40} {gpu:<15} {cost:<10} "
                f"{qual:<6} {price:<6} {lat:<6} {bal:<6}"
            )


def pause(message: str = "Press Enter to continue..."):
    input(f"\n{message}")


def main():
    # ========================================================================
    # Show the starting point
    # ========================================================================
    print_header("Wheel Demo — Planner Library API")
    print("\nThis demo walks through the pipeline using the Planner class")
    print("directly — no server required. Modify DEMO_INTENT at the top")
    print("of this script to try different scenarios.")

    print("\nWe start with a DeploymentIntent:")
    print_json("DeploymentIntent", DEMO_INTENT)

    # ========================================================================
    # Initialize Planner and load benchmarks
    # ========================================================================
    pause("Press Enter to initialize Planner and load benchmarks...")

    print_header("Initialize Planner")
    print("\nCreating Planner instance and loading bundled BLIS benchmarks...")

    p = Planner()
    p.load_bundled_benchmarks()
    print("Done.")

    # ========================================================================
    # Stage 1: generate_specification
    # ========================================================================
    pause("Press Enter to generate specification from intent...")

    print_header("Stage 1: generate_specification(intent)")

    spec = p.generate_specification(DEMO_INTENT)
    print_json("DeploymentSpecification", spec)

    # ========================================================================
    # Stage 2: generate_recommendations
    # ========================================================================
    pause("Press Enter to generate recommendations from specification...")

    print_header("Stage 2: generate_recommendations(spec)")

    recs = p.generate_recommendations(spec)

    print("\nRankedRecommendations (summary):")
    print_recommendation_tables(recs)

    if not recs.balanced:
        print("\nNo recommendations found. Pipeline stopped.")
        sys.exit(0)

    print("\nFor demo purposes, choosing the first balanced recommendation.")

    # ========================================================================
    # Stage 3: generate_deployment
    # ========================================================================
    pause("Press Enter to generate deployment YAML...")

    print_header("Stage 3: generate_deployment(configuration)")

    top = recs.balanced[0]
    config = top.configuration
    print_json("DeploymentConfiguration (input)", config)

    pause("Press Enter to generate YAML from this configuration...")

    bundle = p.generate_deployment(config)

    bundle_summary = {
        "deployment_id": bundle.deployment_id,
        "namespace": bundle.namespace,
        "stack": bundle.stack,
        "files": {k: f"({len(v)} bytes of YAML)" for k, v in bundle.files.items()},
    }
    print_json("DeploymentBundle (summary)", bundle_summary)

    for filename, content in bundle.files.items():
        print(f"\n--- {filename} ---")
        print(content.strip())

    # ========================================================================
    # Done
    # ========================================================================
    print_header("Pipeline Complete")
    print("\nThe full pipeline ran without a server:")
    print("  DeploymentIntent → DeploymentSpecification → RankedRecommendations")
    print("  → DeploymentConfiguration → DeploymentBundle (YAML files)")
    print("\nTo deploy to a cluster, use:")
    print("  p.deploy_bundle_to_cluster(bundle)")
    print("  (requires pip install llm-d-planner[kubernetes])")


if __name__ == "__main__":
    main()
