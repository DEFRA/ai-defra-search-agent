import pytest

from app import config
from app.bedrock import models as bedrock_models
from app.chat import agent, models
from app.models import UnsupportedModelError

# Mock test data
MOCK_QUESTION = "What is the question?"
MOCK_MODEL_ID = "anthropic.claude-3-sonnet"
MOCK_RESPONSE_TEXT_1 = "First response text"
MOCK_RESPONSE_TEXT_2 = "Second response text"

SYSTEM_PROMPT = "You are a DEFRA agent. All communication should be appropriately professional for a UK government service"


@pytest.fixture
def mock_inference_service(mocker):
    """Mock BedrockInferenceService"""
    return mocker.MagicMock()


@pytest.fixture
def mock_config(mocker):
    """Mock app config"""
    mock_config_obj = mocker.MagicMock()
    mock_config_obj.bedrock.available_generation_models = {
        "anthropic.claude-3-sonnet": config.BedrockModelConfig(
            name="anthropic.claude-3-sonnet",
            model_id="anthropic.claude-3-sonnet",
            bedrock_model_id="anthropic.claude-3-sonnet",
            description="A conversational AI model optimized for dialogue.",
            guardrails=None,
        )
    }
    mock_config_obj.bedrock.default_generation_model = MOCK_MODEL_ID
    return mock_config_obj


@pytest.fixture
def bedrock_agent(mock_inference_service, mock_config):
    """BedrockChatAgent with mocked dependencies"""
    return agent.BedrockChatAgent(
        inference_service=mock_inference_service, app_config=mock_config
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_executes_flow_with_correct_parameters(
    bedrock_agent, mock_inference_service, mocker
):
    # Setup
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
        {"type": "text", "text": MOCK_RESPONSE_TEXT_2},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    # Execute
    result = await bedrock_agent.execute_flow(
        question=MOCK_QUESTION, model_id=MOCK_MODEL_ID
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
    assert mock_messages[0]["content"] == [{"text": MOCK_QUESTION}]

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
async def test_handles_single_response_message(
    bedrock_agent, mock_inference_service, mocker
):
    # Setup
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    # Execute
    result = await bedrock_agent.execute_flow(MOCK_QUESTION, model_id=MOCK_MODEL_ID)

    # Assert single message returned
    assert len(result) == 1
    actual_message = result[0]
    assert actual_message.role == "assistant"
    assert actual_message.content == MOCK_RESPONSE_TEXT_1
    assert actual_message.model_id == MOCK_MODEL_ID


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_executes_flow_returns_usage_data(
    bedrock_agent, mock_inference_service, mocker
):
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]
    mock_usage = {
        "input_tokens": 10,
        "output_tokens": 20,
    }
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage=mock_usage,
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    result = await bedrock_agent.execute_flow(MOCK_QUESTION, model_id=MOCK_MODEL_ID)

    assert len(result) == 1
    actual_message = result[0]
    assert actual_message.usage is not None
    assert actual_message.usage.input_tokens == 10
    assert actual_message.usage.output_tokens == 20
    # total_tokens is sum of input + output
    assert actual_message.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_unsupported_model_raises_exception(bedrock_agent):
    unsupported_model_id = "unsupported-model-123"

    with pytest.raises(UnsupportedModelError) as exc_info:
        await bedrock_agent.execute_flow(MOCK_QUESTION, model_id=unsupported_model_id)

    assert (
        str(exc_info.value)
        == f"Requested model '{unsupported_model_id}' is not supported."
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_execute_flow_with_conversation(
    bedrock_agent, mock_inference_service, mocker
):
    conversation = [
        models.UserMessage(
            content="What is Python?",
            model_id=MOCK_MODEL_ID,
            model_name="Claude 3 Sonnet",
        ),
        models.AssistantMessage(
            content="Python is a programming language.",
            model_id=MOCK_MODEL_ID,
            model_name="Claude 3 Sonnet",
            usage=models.TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
        ),
    ]

    # Mock response for the new question
    mock_response_content = [
        {"type": "text", "text": "It was created by Guido van Rossum."},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 20, "output_tokens": 15},
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    # Execute with conversation history
    result = await bedrock_agent.execute_flow(
        question="Who created it?",
        model_id=MOCK_MODEL_ID,
        conversation=conversation,
    )

    # Assert invoke_anthropic was called with the full conversation history
    mock_inference_service.invoke_anthropic.assert_called_once()
    call_args = mock_inference_service.invoke_anthropic.call_args
    messages = call_args[1]["messages"]

    # Should have history (2 messages) + new user message (1) = 3 messages
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": "What is Python?"}]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [{"text": "Python is a programming language."}]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == [{"text": "Who created it?"}]

    # Assert response contains only assistant message (not user message)
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].content == "It was created by Guido van Rossum."


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_execute_flow_without_conversation(
    bedrock_agent, mock_inference_service, mocker
):
    """Test that execute_flow works correctly when no conversation history is provided"""
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    # Execute without conversation history (None)
    result = await bedrock_agent.execute_flow(
        question=MOCK_QUESTION, model_id=MOCK_MODEL_ID, conversation=None
    )

    # Assert invoke_anthropic was called with only the new message
    mock_inference_service.invoke_anthropic.assert_called_once()
    call_args = mock_inference_service.invoke_anthropic.call_args
    messages = call_args[1]["messages"]

    # Should have only the new user message
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": MOCK_QUESTION}]

    # Assert response contains only assistant message
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].content == MOCK_RESPONSE_TEXT_1


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_config")
async def test_execute_flow_with_empty_conversation(
    bedrock_agent, mock_inference_service, mocker
):
    mock_response_content = [
        {"type": "text", "text": MOCK_RESPONSE_TEXT_1},
    ]
    mock_model_response = bedrock_models.ModelResponse(
        model_id=MOCK_MODEL_ID,
        content=mock_response_content,
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    mock_inference_service.invoke_anthropic = mocker.MagicMock(
        return_value=mock_model_response
    )

    # Execute with empty conversation history
    result = await bedrock_agent.execute_flow(
        question=MOCK_QUESTION, model_id=MOCK_MODEL_ID, conversation=[]
    )

    # Assert invoke_anthropic was called with only the new message
    call_args = mock_inference_service.invoke_anthropic.call_args
    messages = call_args[1]["messages"]

    # Should have only the new user message
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": MOCK_QUESTION}]

    # Assert response contains only assistant message
    assert len(result) == 1
    assert result[0].role == "assistant"
