"""Tests for ModelCatalogModelSource."""

from unittest.mock import MagicMock

import pytest

from neuralnav.knowledge_base.model_catalog_models import ModelCatalogModelSource


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {
            "name": "RedHatAI/granite-3.1-8b-instruct",
            "provider": "IBM",
            "description": "Granite model",
            "tasks": ["text-to-text"],
            "license": "Apache 2.0",
            "customProperties": {
                "size": {"string_value": "8B params", "metadataType": "MetadataStringValue"},
                "validated": {"string_value": "", "metadataType": "MetadataStringValue"},
                "validated_on": {
                    "string_value": '["RHOAI 2.20"]',
                    "metadataType": "MetadataStringValue",
                },
            },
        },
    ]
    return client


@pytest.mark.unit
def test_get_all_models(mock_client):
    source = ModelCatalogModelSource(mock_client)
    models = source.get_all_models()
    assert len(models) == 1
    m = models[0]
    assert m.model_id == "RedHatAI/granite-3.1-8b-instruct"
    assert m.provider == "IBM"
    assert m.approval_status == "approved"


@pytest.mark.unit
def test_get_model(mock_client):
    source = ModelCatalogModelSource(mock_client)
    m = source.get_model("RedHatAI/granite-3.1-8b-instruct")
    assert m is not None
    assert m.model_id == "RedHatAI/granite-3.1-8b-instruct"


@pytest.mark.unit
def test_get_model_not_found(mock_client):
    source = ModelCatalogModelSource(mock_client)
    assert source.get_model("unknown/model") is None


@pytest.mark.unit
def test_parse_size_parameters(mock_client):
    source = ModelCatalogModelSource(mock_client)
    models = source.get_all_models()
    assert models[0].size_parameters == "8B"


@pytest.mark.unit
def test_model_fields_complete(mock_client):
    """Verify all ModelInfo fields are populated."""
    source = ModelCatalogModelSource(mock_client)
    m = source.get_all_models()[0]
    assert m.name == "granite-3.1-8b-instruct"
    assert m.family == "granite"
    assert m.license == "Apache 2.0"
    assert m.license_type == "permissive"
    assert m.min_gpu_memory_gb >= 8
    assert m.context_length == 128000
    assert len(m.supported_tasks) > 0
    assert isinstance(m.domain_specialization, list)


@pytest.mark.unit
def test_find_models_for_use_case(mock_client):
    source = ModelCatalogModelSource(mock_client)
    # text-to-text maps to chatbot_conversational among others
    models = source.find_models_for_use_case("chatbot_conversational")
    assert len(models) == 1
    assert models[0].model_id == "RedHatAI/granite-3.1-8b-instruct"


@pytest.mark.unit
def test_find_models_by_task(mock_client):
    source = ModelCatalogModelSource(mock_client)
    models = source.find_models_by_task("chatbot_conversational")
    assert len(models) == 1


@pytest.mark.unit
def test_unapproved_model_excluded(mock_client):
    """Models without 'validated' property are excluded from get_all_models."""
    mock_client.list_models.return_value = [
        {
            "name": "SomeOrg/unvalidated-model",
            "provider": "SomeOrg",
            "tasks": ["text-generation"],
            "license": "proprietary",
            "customProperties": {
                "size": {"string_value": "7B params", "metadataType": "MetadataStringValue"},
            },
        },
    ]
    source = ModelCatalogModelSource(mock_client)
    assert len(source.get_all_models()) == 0
    # But get_model still returns it
    m = source.get_model("SomeOrg/unvalidated-model")
    assert m is not None
    assert m.approval_status == "pending"


@pytest.mark.unit
def test_cache_ttl(mock_client):
    """Second call within TTL does not re-fetch from client."""
    source = ModelCatalogModelSource(mock_client)
    source.get_all_models()
    source.get_all_models()
    # list_models called only once due to caching
    assert mock_client.list_models.call_count == 1


@pytest.mark.unit
def test_size_parsing_variants():
    """Test various size string formats."""
    from neuralnav.knowledge_base.model_catalog_models import _parse_size

    assert _parse_size("8B params") == "8B"
    assert _parse_size("70B params") == "70B"
    assert _parse_size("3.1B") == "3.1B"
    assert _parse_size("0.5b parameters") == "0.5B"
    assert _parse_size("unknown") == "unknown"


@pytest.mark.unit
def test_family_extraction():
    """Test model family extraction from names."""
    from neuralnav.knowledge_base.model_catalog_models import _extract_family

    assert _extract_family("RedHatAI/granite-3.1-8b-instruct") == "granite"
    assert _extract_family("meta-llama/Llama-3.1-70B") == "llama"
    assert _extract_family("mistralai/Mistral-7B-v0.3") == "mistral"
    assert _extract_family("SomeOrg/custom-model-v1") == "custom"
