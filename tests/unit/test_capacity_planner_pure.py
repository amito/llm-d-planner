"""Pure-computation capacity planner tests (no external dependencies)."""

import pytest

from planner.capacity_planner import (
    ACTIVATION_MEMORY_BASE_DENSE_GIB,
    ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB,
    estimate_vllm_activation_memory,
    estimate_vllm_cuda_graph_memory,
    estimate_vllm_non_torch_memory,
    gpus_required,
    parameter_memory_req,
    precision_to_byte,
)


@pytest.mark.unit
def test_precision_to_byte():
    """Tests that precision data type is converted to byte accurately."""
    bytes_8 = ["F64", "I64", "INT64"]
    bytes_4 = ["F32", "I32", "INT32"]
    bytes_2 = ["F16", "BF16", "I16", "INT16"]
    bytes_1 = ["F8_E5M2", "F8_E4M3", "I8", "INT8", "U8"]
    bytes_half = ["FP4", "U4", "I4", "INT4"]
    boolean = ["BOOL"]

    for dtype in bytes_8:
        assert precision_to_byte(dtype) == 8

    for dtype in bytes_4:
        assert precision_to_byte(dtype) == 4

    for dtype in bytes_2:
        assert precision_to_byte(dtype) == 2

    for dtype in bytes_1:
        assert precision_to_byte(dtype) == 1

    for dtype in bytes_half:
        assert precision_to_byte(dtype) == 0.5

    for dtype in boolean:
        assert precision_to_byte(dtype) == 1

    # Special cases
    assert precision_to_byte("f64") == 8
    assert precision_to_byte("ff8_e5m2") == 1


@pytest.mark.unit
def test_parameter_memory_req():
    """Tests parameter memory size is accurately calculated given precision."""
    factor = 1024**3
    params = [10, 1000, 10000, 100000]
    precisions = ["FP32", "FP16", "FP8", "INT4"]
    prec_to_byte = [4, 2, 1, 0.5]

    for param in params:
        for j, precision in enumerate(precisions):
            expected = param * prec_to_byte[j] / factor
            assert parameter_memory_req(param, precision) == expected


@pytest.mark.unit
def test_gpus_required():
    """Tests GPU number required for parallelism is correctly calculated."""
    for tp in range(1, 16):
        for pp in range(1, 16):
            for dp in range(1, 16):
                expected = tp * pp * dp
                assert expected == gpus_required(tp, pp, dp)


@pytest.mark.unit
def test_estimate_vllm_non_torch_memory():
    """Tests that non-torch memory estimation returns TP-dependent values."""
    # TP=1, PP=1: 0.27 GiB (calibrated on vLLM v0.19.0)
    actual_tp1 = estimate_vllm_non_torch_memory(tp=1)
    expected_tp1 = 0.27
    assert actual_tp1 == expected_tp1, f"Expected {expected_tp1} GiB for TP=1, got {actual_tp1} GiB"
    assert isinstance(actual_tp1, float), "Should return a float"

    # TP>=2: 2.10 GiB (NCCL all-reduce buffers)
    actual_tp2 = estimate_vllm_non_torch_memory(tp=2)
    expected_tp2 = 2.10
    assert actual_tp2 == expected_tp2, f"Expected {expected_tp2} GiB for TP=2, got {actual_tp2} GiB"

    actual_tp4 = estimate_vllm_non_torch_memory(tp=4)
    assert actual_tp4 == expected_tp2, f"Expected {expected_tp2} GiB for TP=4, got {actual_tp4} GiB"

    # TP=1, PP>=2: 0.07 GiB (P2P send/receive buffers only)
    actual_pp2 = estimate_vllm_non_torch_memory(tp=1, pp=2)
    expected_pp2 = 0.07
    assert actual_pp2 == expected_pp2, (
        f"Expected {expected_pp2} GiB for TP=1/PP=2, got {actual_pp2} GiB"
    )


@pytest.mark.unit
def test_estimate_vllm_cuda_graph_memory():
    """Tests that CUDA graph memory returns 0.0 (included in activation memory)."""
    expected = 0.0  # CUDA graph memory is included in activation profiling
    actual = estimate_vllm_cuda_graph_memory()
    assert actual == expected, f"Expected {expected} GiB, got {actual} GiB"
    assert isinstance(actual, float), "Should return a float"


@pytest.mark.unit
def test_estimate_vllm_activation_memory_multimodal_fallback():
    """Tests that unknown multimodal architectures fall back to multimodal constant."""

    class FakeMultimodalConfig:
        architectures = ["LlavaForConditionalGeneration"]

    activation = estimate_vllm_activation_memory(FakeMultimodalConfig(), tp=1)  # type: ignore[arg-type]
    assert activation == ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB, (
        f"Unknown multimodal should return {ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB} GiB, got {activation} GiB"
    )


@pytest.mark.unit
def test_estimate_vllm_activation_memory_unknown_dense_fallback():
    """Tests that unknown dense architectures fall back to dense constant."""

    class FakeDenseConfig:
        architectures = ["SomeNewModelForCausalLM"]

    activation = estimate_vllm_activation_memory(FakeDenseConfig(), tp=1)  # type: ignore[arg-type]
    assert activation == ACTIVATION_MEMORY_BASE_DENSE_GIB, (
        f"Unknown dense model should return {ACTIVATION_MEMORY_BASE_DENSE_GIB} GiB, got {activation} GiB"
    )
