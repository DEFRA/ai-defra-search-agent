import json

import pytest

from app import config


def test_get_config(bedrock_generation_models):
    config_obj = config.get_config()
    assert config_obj is not None
    assert config_obj.mongo.uri == "mongodb://dummy:27017/test_db"

    expected_models = {}
    for model in bedrock_generation_models:
        guardrails = None
        if "guardrails" in model:
            guardrails = config.BedrockGuardrailConfig(
                guardrail_id=model["guardrails"]["guardrail_id"],
                guardrail_version=model["guardrails"]["guardrail_version"],
            )

        expected_models[model["modelId"]] = config.BedrockModelConfig(
            name=model["name"],
            bedrock_model_id=model["bedrockModelId"],
            model_id=model["modelId"],
            description=model["description"],
            guardrails=guardrails,
        )

    assert config_obj.bedrock.available_generation_models == expected_models


def test_get_config_raises_runtime_error_when_invalid_config(monkeypatch):
    monkeypatch.setenv("AWS_BEDROCK_AVAILABLE_GENERATION_MODELS", "invalid_config")
    with pytest.raises(RuntimeError, match="Config validation failed with errors:.*"):
        config.get_config()


def test_get_config_raises_error_on_duplicate_model_ids(monkeypatch):
    duplicate_models = [
        {
            "name": "Model 1",
            "modelId": "duplicate-id",
            "bedrockModelId": "arn:model1",
            "description": "First model",
        },
        {
            "name": "Model 2",
            "modelId": "duplicate-id",
            "bedrockModelId": "arn:model2",
            "description": "Second model",
        },
    ]
    monkeypatch.setenv(
        "AWS_BEDROCK_AVAILABLE_GENERATION_MODELS", json.dumps(duplicate_models)
    )

    with pytest.raises(RuntimeError) as excinfo:
        config.get_config()

    assert "Duplicate model id found in configuration: duplicate-id" in str(
        excinfo.value
    )


def test_mongo_config_defaults():
    mongo_config = config.MongoConfig()
    assert mongo_config.server_selection_timeout_ms == 5000
    assert mongo_config.connect_timeout_ms == 5000
    assert mongo_config.socket_timeout_ms == 10000
    assert mongo_config.retry_attempts == 2
    assert mongo_config.retry_base_delay_seconds == 0.5


def test_mongo_config_env_var_overrides(monkeypatch):
    monkeypatch.setenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "3000")
    monkeypatch.setenv("MONGO_CONNECT_TIMEOUT_MS", "2000")
    monkeypatch.setenv("MONGO_SOCKET_TIMEOUT_MS", "8000")
    monkeypatch.setenv("MONGO_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("MONGO_RETRY_BASE_DELAY_SECONDS", "1.5")

    mongo_config = config.MongoConfig()
    assert mongo_config.server_selection_timeout_ms == 3000
    assert mongo_config.connect_timeout_ms == 2000
    assert mongo_config.socket_timeout_ms == 8000
    assert mongo_config.retry_attempts == 4
    assert mongo_config.retry_base_delay_seconds == 1.5


def test_knowledge_config_loads_without_knowledge_group_id(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_URL", "http://knowledge-service:8087")
    knowledge_config = config.KnowledgeConfig()
    assert knowledge_config.base_url == "http://knowledge-service:8087"
    assert not hasattr(knowledge_config, "knowledge_group_id")
