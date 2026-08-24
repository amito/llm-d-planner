#!/usr/bin/env python3
"""Unit and HuggingFace-network tests for CostManager and GPURecommender cost features."""

import pytest

from planner.gpu_recommender import CostManager, GPURecommender


@pytest.mark.unit
def test_cost_manager():
    """Test CostManager functionality"""
    print("=" * 80)
    print("Testing CostManager")
    print("=" * 80)

    # Test default costs
    cm = CostManager()
    print(f"✅ CostManager initialized with {len(cm.default_costs)} GPUs")

    # Test getting costs
    h100_cost = cm.get_cost("H100")
    print(f"✅ H100 cost: ${h100_cost}")

    a100_cost = cm.get_cost("A100")
    print(f"✅ A100 cost: ${a100_cost}")

    # Test multi-GPU cost
    h100_2gpu = cm.get_cost("H100", num_gpus=2)
    print(f"✅ H100 (2 GPUs) cost: ${h100_2gpu}")
    assert h100_cost is not None
    assert h100_2gpu == h100_cost * 2, "Multi-GPU cost calculation failed"

    # Test custom costs
    custom_costs = {"H100": 30.0, "A100": 20.0}
    cm_custom = CostManager(custom_costs=custom_costs)
    h100_custom = cm_custom.get_cost("H100")
    print(f"✅ H100 custom cost: ${h100_custom}")
    assert h100_custom == 30.0, "Custom cost override failed"

    print("\n✅ All CostManager tests passed!\n")


@pytest.mark.hf_network
def test_gpu_recommender():
    """Test GPURecommender cost integration"""
    print("=" * 80)
    print("Testing GPURecommender Cost Integration")
    print("=" * 80)

    # Test basic initialization
    recommender = GPURecommender(
        model_id="Qwen/Qwen-7B",
        input_len=512,
        output_len=128,
        max_gpus=1,
        gpu_list=["H100", "A100"],
    )
    print("✅ GPURecommender initialized")

    # Test cost manager is available
    assert recommender.cost_manager is not None, "CostManager not initialized"
    print("✅ CostManager integrated into GPURecommender")

    # Test custom costs
    custom_costs = {"H100": 30.0, "A100": 20.0}
    recommender_custom = GPURecommender(
        model_id="Qwen/Qwen-7B",
        input_len=512,
        output_len=128,
        max_gpus=1,
        gpu_list=["H100", "A100"],
        custom_gpu_costs=custom_costs,
    )
    print("✅ GPURecommender with custom costs initialized")

    # Verify custom costs are set
    h100_cost = recommender_custom.cost_manager.get_cost("H100")
    assert h100_cost == 30.0, "Custom costs not applied"
    print(f"✅ Custom costs applied correctly: H100 = ${h100_cost}")

    # Test methods exist
    assert hasattr(recommender, "get_gpu_with_lowest_cost"), (
        "Missing get_gpu_with_lowest_cost method"
    )
    assert hasattr(recommender, "get_results_sorted_by_cost"), (
        "Missing get_results_sorted_by_cost method"
    )
    print("✅ New cost methods available")

    print("\n✅ All GPURecommender integration tests passed!\n")
