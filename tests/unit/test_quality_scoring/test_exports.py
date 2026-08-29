"""Verify all public symbols are importable from quality_scoring directly."""

import pytest


@pytest.mark.unit
def test_all_public_exports_importable():
    import quality_scoring

    expected = [
        "ALL_CATEGORIES",
        "CATEGORY_GROUPS",
        "CategoryFinding",
        "CompositeScore",
        "DEFAULT_CATEGORIES",
        "DISPLAY_NAMES",
        "MatchType",
        "ModelScorecard",
        "NormalizedScore",
        "ScoringEngine",
        "display_name",
        "generate_category_findings",
        "suggest_similar",
    ]

    for name in expected:
        assert hasattr(quality_scoring, name), f"{name} not found in quality_scoring"
        assert name in quality_scoring.__all__, f"{name} not in quality_scoring.__all__"


@pytest.mark.unit
def test_new_exports_are_correct_types():
    from quality_scoring import (
        ALL_CATEGORIES,
        CATEGORY_GROUPS,
        DEFAULT_CATEGORIES,
        DISPLAY_NAMES,
        CategoryFinding,
        display_name,
        generate_category_findings,
        suggest_similar,
    )

    assert isinstance(ALL_CATEGORIES, list)
    assert isinstance(DEFAULT_CATEGORIES, list)
    assert isinstance(DISPLAY_NAMES, dict)
    assert isinstance(CATEGORY_GROUPS, list)
    assert "overall" in ALL_CATEGORIES
    assert "overall" in DEFAULT_CATEGORIES
    assert "overall" in DISPLAY_NAMES
    assert callable(display_name)
    assert callable(generate_category_findings)
    assert callable(suggest_similar)
    assert display_name("overall") == "Overall"
    assert issubclass(CategoryFinding, object)
