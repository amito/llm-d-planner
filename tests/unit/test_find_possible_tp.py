"""Tests for find_possible_tp() vLLM-compatible TP validation."""

from unittest.mock import MagicMock

import pytest

from planner.capacity_planner import find_possible_tp


def _make_config(
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    intermediate_size: int = 11008,
    vocab_size: int = 32000,
) -> MagicMock:
    """Build a minimal model config matching HuggingFace AutoConfig layout."""
    cfg = MagicMock()
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = (
        num_attention_heads if num_key_value_heads is None else num_key_value_heads
    )
    cfg.intermediate_size = intermediate_size
    cfg.vocab_size = vocab_size
    cfg.text_config = cfg  # so get_text_config returns cfg itself
    return cfg


@pytest.mark.unit
def test_qwen25_05b_only_1_and_2():
    """Qwen2.5-0.5B: 14 heads, 2 kv_heads, intermediate=4864, vocab=151936."""
    cfg = _make_config(14, num_key_value_heads=2, intermediate_size=4864, vocab_size=151936)
    assert find_possible_tp(cfg) == [1, 2]


@pytest.mark.unit
def test_llama_7b_like():
    """Typical Llama-7B: 32 heads, 8 kv_heads, intermediate=11008, vocab=32000."""
    cfg = _make_config(32, num_key_value_heads=8, intermediate_size=11008, vocab_size=32000)
    assert find_possible_tp(cfg) == [1, 2, 4, 8, 16, 32]


@pytest.mark.unit
def test_40_heads_llama_70b_like():
    """Llama-70B-like: 40 heads, 8 kv_heads, intermediate=13824, vocab=32000."""
    cfg = _make_config(40, num_key_value_heads=8, intermediate_size=13824, vocab_size=32000)
    assert find_possible_tp(cfg) == [1, 2, 4, 8]


@pytest.mark.unit
def test_28_heads_filtered_by_intermediate():
    """28 heads with intermediate_size=4096: 7 and 14 fail intermediate check."""
    cfg = _make_config(28, num_key_value_heads=4, intermediate_size=4096, vocab_size=32000)
    assert find_possible_tp(cfg) == [1, 2, 4]


@pytest.mark.unit
def test_non_power_of_2_tp_allowed():
    """36 heads with compatible sizes: TP=12 is valid (not power of 2)."""
    cfg = _make_config(36, num_key_value_heads=4, intermediate_size=9216, vocab_size=32064)
    assert find_possible_tp(cfg) == [1, 2, 4, 12]


@pytest.mark.unit
def test_single_head():
    """Single attention head: only TP=1 is valid."""
    cfg = _make_config(1, num_key_value_heads=1, intermediate_size=2048, vocab_size=32000)
    assert find_possible_tp(cfg) == [1]


@pytest.mark.unit
def test_64_heads():
    """64 heads, 8 kv_heads, all power-of-2 TPs valid."""
    cfg = _make_config(64, num_key_value_heads=8, intermediate_size=16384, vocab_size=128256)
    assert find_possible_tp(cfg) == [1, 2, 4, 8, 16, 32, 64]


@pytest.mark.unit
def test_kv_heads_gqa_branch():
    """When tp > num_kv_heads, vLLM checks tp % kv_heads == 0 instead."""
    # 16 heads, 2 kv_heads: TP=4 works because 4 % 2 == 0 (GQA branch)
    cfg = _make_config(16, num_key_value_heads=2, intermediate_size=8192, vocab_size=32000)
    result = find_possible_tp(cfg)
    assert 4 in result
    assert 8 in result


@pytest.mark.unit
def test_vocab_size_rejects_tp():
    """padded_vocab_size (100→128) is not divisible by 3 or 6, rejecting those TPs."""
    cfg = _make_config(6, num_key_value_heads=6, intermediate_size=6, vocab_size=100)
    result = find_possible_tp(cfg)
    assert result == [1, 2]
    assert 3 not in result
    assert 6 not in result


@pytest.mark.unit
def test_kv_heads_default_to_attention_heads():
    """When num_key_value_heads is absent, it defaults to num_attention_heads (MHA)."""
    cfg = MagicMock()
    cfg.num_attention_heads = 32
    cfg.intermediate_size = 11008
    cfg.vocab_size = 32000
    cfg.text_config = cfg
    # No num_key_value_heads attribute — getattr fallback
    del cfg.num_key_value_heads
    assert find_possible_tp(cfg) == [1, 2, 4, 8, 16, 32]
