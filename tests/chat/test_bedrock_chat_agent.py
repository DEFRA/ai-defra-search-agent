from unittest.mock import MagicMock

import pytest

from app import config
from app.bedrock import models as bedrock_models
from app.chat import agent, models

# Mock test data
MOCK_QUESTION = "What is the question?"
MOCK_MODEL_ID = "anthropic.claude-3-sonnet"
MOCK_RESPONSE_TEXT_1 = "First response text"
MOCK_RESPONSE_TEXT_2 = "Second response text"

SYSTEM_PROMPT = "You are a DEFRA agent. All communication should be appropriately professional for a UK government service"


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
async def test_executes_flow_with_correct_parameters(
    bedrock_agent, mock_inference_service
):
    # Setup
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
        {"type": "text", "text": MOCK_RESPONSE_TEXT_2},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
    )
    mock_inference_service.invoke_anthropic = MagicMock(
        return_value=mock_model_response
    )

    # Execute
    result = await bedrock_agent.execute_flow(
        question=MOCK_QUESTION, model_name=MOCK_MODEL_ID
    )

    # Assert invoke_anthropic called with correct parameters
    mock_inference_service.invoke_anthropic.assert_called_once()
    call_args = mock_inference_service.invoke_anthropic.call_args

    # Assert model parameter
    assert call_args[1]["model_config"].id == MOCK_MODEL_ID

    # Assert system prompt parameter
    assert call_args[1]["system_prompt"] == SYSTEM_PROMPT

    # Assert messages parameter
    mock_messages = call_args[1]["messages"]
    assert len(mock_messages) == 1
    assert mock_messages[0]["role"] == "user"
    assert mock_messages[0]["content"] == MOCK_QUESTION

    # Assert response converted to messages correctly
    assert len(result) == 2  # 2 response messages

    # Verify each message matches the mock response content
    for i, mock_content in enumerate(mock_response_content):
        actual_message = result[i]
        assert actual_message.role == "assistant"
        assert actual_message.content == mock_content["text"]
        assert actual_message.model_id == MOCK_MODEL_ID


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_handles_single_response_message(bedrock_agent, mock_inference_service):
    # Setup
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
    )
    mock_inference_service.invoke_anthropic = MagicMock(
        return_value=mock_model_response
    )

    # Execute
    result = await bedrock_agent.execute_flow(MOCK_QUESTION, model_name=MOCK_MODEL_ID)

    # Assert single message returned
    assert len(result) == 1
    actual_message = result[0]
    assert actual_message.role == "assistant"
    assert actual_message.content == MOCK_RESPONSE_TEXT_1
    assert actual_message.model_id == MOCK_MODEL_ID


@pytest.mark.asyncio
async def test_unsupported_model_raises_exception(bedrock_agent):
    unsupported_model_id = "unsupported-model-123"

    with pytest.raises(models.UnsupportedModelError) as exc_info:
        await bedrock_agent.execute_flow(MOCK_QUESTION, model_name=unsupported_model_id)

    assert (
        str(exc_info.value)
        == f"Requested model '{unsupported_model_id}' is not supported."
    )
