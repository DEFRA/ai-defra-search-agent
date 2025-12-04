from unittest.mock import MagicMock

import pytest

from app import config
from app.bedrock import models as bedrock_models
from app.chat import agent

# Mock test data
MOCK_QUESTION = "What is the question?"
MOCK_MODEL_ID = "anthropic.claude-3-sonnet"
MOCK_RESPONSE_TEXT_1 = "First response text"


@pytest.fixture
def mock_inference_service():
    """Mock BedrockInferenceService"""
    return MagicMock()


@pytest.fixture
def mock_config(monkeypatch):
    """Mock app config"""
    mock_config_obj = MagicMock()
    mock_config_obj.bedrock.available_generation_models = {
        "anthropic.claude-3-sonnet": config.BedrockModelConfig(
            name="anthropic.claude-3-sonnet",
            id="anthropic.claude-3-sonnet",
            description="A conversational AI model optimized for dialogue.",
            guardrails=None,
        )
    }
    mock_config_obj.bedrock.default_generation_model = MOCK_MODEL_ID
    monkeypatch.setattr("app.chat.agent.app_config", mock_config_obj)
    return mock_config_obj


@pytest.fixture
def bedrock_agent(mock_inference_service):
    """BedrockChatAgent with mocked dependencies"""
    return agent.BedrockChatAgent(inference_service=mock_inference_service)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_raises_type_error_on_missing_usage_data(
    bedrock_agent, mock_inference_service
):
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]

    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage=None,
    )
    mock_inference_service.invoke_anthropic = MagicMock(
        return_value=mock_model_response
    )

    with pytest.raises(TypeError):
        await bedrock_agent.execute_flow(MOCK_QUESTION, model_name=MOCK_MODEL_ID)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_raises_key_error_on_partial_usage_data(
    bedrock_agent, mock_inference_service
):
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]

    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 15},
    )
    mock_inference_service.invoke_anthropic = MagicMock(
        return_value=mock_model_response
    )

    with pytest.raises(KeyError) as excinfo:
        await bedrock_agent.execute_flow(MOCK_QUESTION, model_name=MOCK_MODEL_ID)
    assert "'output_tokens'" in str(excinfo.value)
