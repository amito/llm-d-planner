"""Tests for ModelCatalogQualityScorer."""

from unittest.mock import MagicMock

import pytest

from neuralnav.knowledge_base.model_catalog_quality import ModelCatalogQualityScorer


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {"name": "RedHatAI/model-a"},
        {"name": "RedHatAI/model-b"},
    ]
    client.get_model_artifacts.side_effect = lambda name: {
        "RedHatAI/model-a": [
            {
                "artifactType": "metrics-artifact",
                "metricsType": "accuracy-metrics",
                "customProperties": {
                    "overall_average": {
                        "double_value": 57.67,
                        "metadataType": "MetadataDoubleValue",
                    },
                    "mmlu": {
                        "double_value": 82.04,
                        "metadataType": "MetadataDoubleValue",
                    },
                },
            }
        ],
        "RedHatAI/model-b": [
            {
                "artifactType": "metrics-artifact",
                "metricsType": "accuracy-metrics",
                "customProperties": {
                    "overall_average": {
                        "double_value": 42.0,
                        "metadataType": "MetadataDoubleValue",
                    },
                },
            }
        ],
    }.get(name, [])
    return client


@pytest.mark.unit
def test_get_quality_score(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("RedHatAI/model-a", "chatbot_conversational")
    assert score == pytest.approx(57.67)


@pytest.mark.unit
def test_get_quality_score_unknown_model(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("unknown/model", "chatbot_conversational")
    assert score == 0.0


@pytest.mark.unit
def test_get_quality_score_case_insensitive(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("redhatai/model-a", "chatbot_conversational")
    assert score == pytest.approx(57.67)
