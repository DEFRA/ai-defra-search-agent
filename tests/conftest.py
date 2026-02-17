import json

import pytest

from app import config

pytest_plugins = ["tests.fixtures.bedrock", "tests.fixtures.mongo"]


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

    # Use a dummy URI by default so we don't spin up Docker for every test
    monkeypatch.setenv("MONGO_URI", "mongodb://dummy:27017/test_db")
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
    monkeypatch.setenv("KNOWLEDGE_BASE_URL", "http://knowledge-base.com")
    monkeypatch.setenv(
        "SQS_CHAT_QUEUE_URL",
        "http://sqs.eu-central-1.localstack:4566/000000000000/chat-job-queue",
    )
    monkeypatch.setenv("KNOWLEDGE_GROUP_ID", "kg-1234567890")
    monkeypatch.setenv("KNOWLEDGE_SIMILARITY_THRESHOLD", "0.5")
