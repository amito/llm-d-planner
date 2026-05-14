"""Test UseCaseQualityScorer catalog fallback and benchmark mapping."""

import pytest

from planner.recommendation.quality.usecase_scorer import UseCaseQualityScorer


@pytest.mark.unit
def test_set_catalog_fallback_provides_score():
    scorer = UseCaseQualityScorer()
    scorer.set_catalog_fallback({"catalog/only-model": 72.5})
    score = scorer.get_quality_score("catalog/only-model", "chatbot_conversational")
    assert score == pytest.approx(72.5)


@pytest.mark.unit
def test_csv_score_takes_precedence_over_fallback():
    """If a model has a CSV score, the fallback is not used."""
    scorer = UseCaseQualityScorer()
    # Pick a model that exists in the CSV data
    csv_score = scorer.get_quality_score("granite 3.3 8b (non-reasoning)", "chatbot_conversational")
    # Now set a fallback with a different score
    scorer.set_catalog_fallback({"granite 3.3 8b (non-reasoning)": 99.9})
    after = scorer.get_quality_score("granite 3.3 8b (non-reasoning)", "chatbot_conversational")
    assert after == csv_score  # CSV wins


@pytest.mark.unit
def test_fallback_returns_zero_for_unknown_model():
    scorer = UseCaseQualityScorer()
    scorer.set_catalog_fallback({"some/model": 50.0})
    score = scorer.get_quality_score("completely/unknown", "chatbot_conversational")
    assert score == 0.0


@pytest.mark.unit
def test_normalize_strips_new_suffixes():
    scorer = UseCaseQualityScorer()
    assert scorer._normalize_model_name("RedHatAI/Phi-4-reasoning-FP8-dynamic") == "phi-4"
    assert (
        scorer._normalize_model_name("mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4")
        == "mistral-large-3-675b"
    )
    assert (
        scorer._normalize_model_name("RedHatAI/Apertus-8B-Instruct-2509-FP8-dynamic")
        == "apertus-8b"
    )
    assert (
        scorer._normalize_model_name("RedHatAI/Ministral-3-14B-Instruct-2512") == "ministral-3-14b"
    )


@pytest.mark.unit
def test_map_resolves_qwen25_correctly():
    """Qwen2.5-7B should map to 'qwen2.5 max', not 'qwen2.5 coder instruct 7b'."""
    scorer = UseCaseQualityScorer()
    score = scorer.get_quality_score("Qwen/Qwen2.5-7B-Instruct", "chatbot_conversational")
    assert score > 30  # qwen2.5 max is ~37%, coder was only 20%


@pytest.mark.unit
def test_quantization_discount_applied():
    """W4A16 should score lower than base model; FP8 should be the same."""
    scorer = UseCaseQualityScorer()
    base = scorer.get_quality_score("Qwen/Qwen2.5-7B-Instruct", "chatbot_conversational")
    fp8 = scorer.get_quality_score(
        "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic", "chatbot_conversational"
    )
    w4a16 = scorer.get_quality_score(
        "RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16", "chatbot_conversational"
    )
    assert fp8 == base  # FP8 discount is 1.0
    assert w4a16 < base  # W4A16 gets 0.92 discount
    assert w4a16 > base * 0.9  # But not too large a discount


@pytest.mark.unit
def test_map_resolves_mistral_small_correctly():
    """Mistral Small 3.1 should map to 'mistral small 3.1', not deephermes."""
    scorer = UseCaseQualityScorer()
    score = scorer.get_quality_score(
        "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",
        "chatbot_conversational",
    )
    assert score > 25  # mistral small 3.1 is ~29%, deephermes was 22%


@pytest.mark.unit
def test_map_resolves_kimi_k2_not_thinking():
    """Kimi K2 should map to 'kimi k2' (55%), not 'kimi k2 thinking' (76%)."""
    scorer = UseCaseQualityScorer()
    score = scorer.get_quality_score(
        "RedHatAI/Kimi-K2-Instruct-quantized.w4a16",
        "chatbot_conversational",
    )
    assert 50 < score < 60


@pytest.mark.unit
def test_new_models_get_scores():
    """New benchmark models should resolve to non-zero scores."""
    scorer = UseCaseQualityScorer()
    assert (
        scorer.get_quality_score(
            "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16", "chatbot_conversational"
        )
        > 50
    )
    assert (
        scorer.get_quality_score(
            "RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4", "chatbot_conversational"
        )
        > 50
    )
    assert (
        scorer.get_quality_score(
            "mistralai/Mistral-Large-3-675B-Instruct-2512", "chatbot_conversational"
        )
        > 25
    )


@pytest.mark.unit
def test_models_with_no_match_return_zero():
    """Apertus and SmolLM3 have no quality data — should return 0."""
    scorer = UseCaseQualityScorer()
    assert (
        scorer.get_quality_score(
            "RedHatAI/Apertus-8B-Instruct-2509-FP8-dynamic", "chatbot_conversational"
        )
        == 0.0
    )
    assert (
        scorer.get_quality_score("RedHatAI/SmolLM3-3B-FP8-dynamic", "chatbot_conversational") == 0.0
    )
