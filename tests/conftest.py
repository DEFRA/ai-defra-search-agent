import json

import pytest

from app import config

pytest_plugins = ["tests.fixtures.bedrock"]


# Reset the global app config variable before each test
@pytest.fixture(autouse=True)
def reset_app_config():
    config.config = None
    yield
    config.config = None


@pytest.fixture(scope="session")
def bedrock_generation_models():
    return [
        {
            "name": "Geni AI 3.5",
            "modelId": "geni-ai-3.5",
            "bedrockModelId": "arn:geni-ai-3.5",
            "description": "Test model 3.5",
        },
        {
            "name": "Geni AI 4",
            "modelId": "geni-ai-4",
            "bedrockModelId": "arn:geni-ai-4",
            "description": "Test model 4",
            "guardrails": {
                "guardrail_id": "arn:aws:bedrock:eu-west-2:123456789012:guardrail/x1234567",
                "guardrail_version": "1",
            },
        },
    ]


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, bedrock_generation_models):
    """Set environment variables for the test session."""
    monkeypatch.setenv("MONGO_URI", "mongodb://mongodb:27018/test_db")
    monkeypatch.setenv(
        "AWS_BEDROCK_AVAILABLE_GENERATION_MODELS", json.dumps(bedrock_generation_models)
    )
    monkeypatch.setenv("AWS_BEDROCK_DEFAULT_GENERATION_MODEL", "Geni AI 3.5")
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("LOG_CONFIG", "logging.json")
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("AWS_BEDROCK_USE_CREDENTIALS", "False")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")  # noqa: S105

    return
