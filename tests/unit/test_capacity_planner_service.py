"""Unit tests for capacity_planner service functions."""

from unittest.mock import MagicMock, patch

import pytest

MOCK_PATH = "planner.capacity_planner"


def _mock_config(
    arch="LlamaForCausalLM",
    is_moe=False,
    is_multimodal=False,
    is_quantized=False,
):
    cfg = MagicMock()
    cfg.architectures = [arch]
    cfg.num_hidden_layers = 32
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 8
    cfg.hidden_size = 4096
    cfg.text_config = cfg

    # For quantization detection: delete quantization_config if not quantized
    # is_quantized() uses hasattr() which returns True for MagicMock attributes by default
    if not is_quantized:
        if hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")
    else:
        cfg.quantization_config = MagicMock()

    # For MoE detection: delete all MoE attributes if is_moe=False
    # is_moe() uses hasattr() which returns True for MagicMock attributes by default
    if not is_moe:
        for attr in ["n_routed_experts", "n_shared_experts", "num_experts", "num_experts_per_tok"]:
            if hasattr(cfg, attr):
                delattr(cfg, attr)
    else:
        # For MoE models, add at least one indicator
        cfg.num_experts = 8
    return cfg


# ---------------------------------------------------------------------------
# get_model_info_summary tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch(f"{MOCK_PATH}.model_memory_req", return_value=14.0)
@patch(f"{MOCK_PATH}.model_params_by_dtype", return_value={"BF16": 7_000_000_000})
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4])
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_get_model_info_summary_dense(mock_config, mock_tp, mock_params, mock_mem):
    from planner.capacity_planner import get_model_info_summary

    mock_config.return_value = _mock_config()
    result = get_model_info_summary("meta-llama/Llama-3-8B", hf_token=None)

    assert result["success"] is True
    assert result["model_id"] == "meta-llama/Llama-3-8B"
    assert result["model_memory_gb"] == 14.0
    assert result["possible_tp_values"] == [1, 2, 4]
    assert result["architecture"]["model_type"] == "Dense"
    assert result["architecture"]["architecture_name"] == "LlamaForCausalLM"
    assert result["architecture"]["is_moe"] is False
    assert result["architecture"]["is_multimodal"] is False
    assert result["quantization"]["is_quantized"] is False
    assert result["activation_memory"]["activation_memory_gb"] > 0
    assert "validated_profiles" in result["activation_memory"]
    assert "base_constants" in result["activation_memory"]
    assert isinstance(result["memory_breakdown"], list)


@pytest.mark.unit
@patch(f"{MOCK_PATH}.model_memory_req", return_value=80.0)
@patch(f"{MOCK_PATH}.model_params_by_dtype", return_value={"BF16": 40_000_000_000})
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4, 8])
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_get_model_info_summary_moe(mock_config, mock_tp, mock_params, mock_mem):
    from planner.capacity_planner import get_model_info_summary

    mock_config.return_value = _mock_config(is_moe=True)
    result = get_model_info_summary("deepseek-ai/DeepSeek-V3", hf_token=None)

    assert result["architecture"]["model_type"] == "MoE"
    assert result["architecture"]["is_moe"] is True


@pytest.mark.unit
@patch(f"{MOCK_PATH}.get_model_config_from_hf", side_effect=Exception("gated repo"))
def test_get_model_info_summary_hf_error_propagates(mock_config):
    """Service function must NOT swallow HF errors — the route maps them to HTTP."""
    from planner.capacity_planner import get_model_info_summary

    with pytest.raises(Exception, match="gated repo"):
        get_model_info_summary("meta-llama/gated-model")
