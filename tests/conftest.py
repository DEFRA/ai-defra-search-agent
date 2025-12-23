import os

import pytest

from app import config


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set environment variables for the test session."""
    env_vars = {
        "MONGO_URI": "mongodb://mongodb:27018/test_db",
        "AWS_BEDROCK_AVAILABLE_GENERATION_MODELS": (
            '[{"name": "Geni AI 3.5", "modelId": "geni-ai-3.5", "bedrockModelId": "arn:geni-ai-3.5", "description": "Test model 3.5."}, {"name": "Geni AI 4", "modelId": "geni-ai-4", "bedrockModelId": "arn:geni-ai-4", "description": "Test model 4"}]'
        ),
        "AWS_BEDROCK_DEFAULT_GENERATION_MODEL": "Geni AI 3.5",
        "PORT": "8000",
        "LOG_CONFIG": "logging.json",
        "AWS_REGION": "eu-west-2",
        "AWS_BEDROCK_USE_CREDENTIALS": "False",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",  # noqa: S105
        "AWS_DEFAULT_REGION": "eu-west-2",
    }

    original_env = dict(os.environ)
    os.environ.update(env_vars)

    # Force reload of config to pick up new env vars if it was already instantiated
    config.config = None
    yield

    # Restore original environment and reset config
    os.environ.clear()
    os.environ.update(original_env)
    config.config = None
