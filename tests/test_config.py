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
