import json
from pathlib import Path

import pytest

from app import config


def test_get_config(bedrock_generation_models, mongo_uri):
    config_obj = config.get_config()
    assert config_obj is not None
    assert config_obj.mongo.uri == mongo_uri

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


def test_system_prompt_loads_from_file():
    config_obj = config.get_config()

    # Read the expected content from the file
    prompt_path = (
        Path(__file__).parent.parent
        / "app"
        / "resources"
        / "prompts"
        / "system_prompt.txt"
    )
    with open(prompt_path, encoding="utf-8") as f:
        expected_prompt = f.read().strip()

    # Verify the system_prompt property returns the file content
    assert config_obj.system_prompt == expected_prompt
    assert len(config_obj.system_prompt) > 0


def test_system_prompt_is_cached():
    config_obj = config.get_config()

    # Access system_prompt twice and verify it returns the same object (cached)
    first_access = config_obj.system_prompt
    second_access = config_obj.system_prompt

    # Should be the exact same object (cached)
    assert first_access is second_access
