import uuid
from unittest.mock import AsyncMock

import pytest

from app.chat import agent, models, repository, service

MOCK_QUESTION = "What is the question?"
MOCK_RESPONSE_1 = "First response"
MOCK_RESPONSE_2 = "Second response"
MOCK_RESPONSE_3 = "Third response"
MOCK_RESPONSE_4 = "Fourth response"
MOCK_PRIOR_MESSAGE = "Previous question"
MOCK_USAGE = models.TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20)


@pytest.fixture
def mock_agent():
    """Async mock of AbstractChatAgent"""
    return AsyncMock(spec=agent.AbstractChatAgent)


@pytest.fixture
def mock_repository():
    """Async mock of AbstractConversationRepository"""
    return AsyncMock(spec=repository.AbstractConversationRepository)


@pytest.fixture
def chat_service(mock_agent, mock_repository):
    """ChatService instance with mocked dependencies"""
    return service.ChatService(
        chat_agent=mock_agent,
        conversation_repository=mock_repository,
    )


@pytest.fixture
def mock_existing_conversation():
    """Conversation with prior messages"""
    prior_messages = [models.UserMessage(content=MOCK_PRIOR_MESSAGE, model_id="gpt-4")]
    return models.Conversation(id=str(uuid.uuid4()), messages=prior_messages.copy())


@pytest.mark.asyncio
async def test_executes_with_existing_conversation(
    chat_service, mock_agent, mock_repository, mock_existing_conversation
):
    # Setup
    mock_agent_responses = [
        models.AssistantMessage(
            content=MOCK_RESPONSE_1, usage=MOCK_USAGE, model_id="gpt-4"
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_2, usage=MOCK_USAGE, model_id="gpt-4"
        ),
    ]
    mock_agent.execute_flow.return_value = mock_agent_responses
    mock_repository.get.return_value = mock_existing_conversation

    # Execute
    result = await chat_service.execute_chat(
        MOCK_QUESTION, "Geni AI-3.5", mock_existing_conversation.id
    )

    # Assert repository.get called correctly
    mock_repository.get.assert_called_once_with(mock_existing_conversation.id)

    # Assert agent called with question string and model name
    mock_agent.execute_flow.assert_called_once_with(
        question=MOCK_QUESTION, model_name="Geni AI-3.5"
    )

    # Assert user message added
    assert len(result.messages) == 4  # 1 prior + 1 user + 2 agent
    assert result.messages[1].role == "user"
    assert result.messages[1].content == MOCK_QUESTION

    # Assert agent responses added (use the mock data)
    for i, mock_response in enumerate(mock_agent_responses):
        actual_message = result.messages[i + 2]
        assert actual_message.role == mock_response.role
        assert actual_message.content == mock_response.content

    # Assert repository.save called with final conversation
    mock_repository.save.assert_called_once()
    saved_conversation = mock_repository.save.call_args[0][0]
    assert saved_conversation.id == mock_existing_conversation.id
    assert len(saved_conversation.messages) == 4

    # Assert conversation ID preserved
    assert result.id == mock_existing_conversation.id


@pytest.mark.asyncio
async def test_creates_new_conversation_when_none_provided(
    chat_service, mock_agent, mock_repository
):
    # Setup
    mock_agent_responses = [
        models.AssistantMessage(
            content=MOCK_RESPONSE_1, usage=MOCK_USAGE, model_id="gpt-4"
        ),
    ]
    mock_agent.execute_flow.return_value = mock_agent_responses

    # Execute - no conversation_id provided
    result = await chat_service.execute_chat(MOCK_QUESTION, "Geni AI-3.5")

    # Assert repository.get NOT called
    mock_repository.get.assert_not_called()

    # Assert agent called with question string and model name
    mock_agent.execute_flow.assert_called_once_with(
        question=MOCK_QUESTION, model_name="Geni AI-3.5"
    )

    # Assert new conversation created
    assert result.id is not None
    assert len(result.messages) == 2  # 1 user + 1 agent

    # Assert user message added
    assert result.messages[0].role == "user"
    assert result.messages[0].content == MOCK_QUESTION

    # Assert agent response added (use the mock data)
    for i, mock_response in enumerate(mock_agent_responses):
        actual_message = result.messages[i + 1]
        assert actual_message.role == mock_response.role
        assert actual_message.content == mock_response.content

    # Assert repository.save called with new conversation
    mock_repository.save.assert_called_once()
    saved_conversation = mock_repository.save.call_args[0][0]
    assert saved_conversation.id == result.id
    assert len(saved_conversation.messages) == 2


@pytest.mark.asyncio
async def test_raises_when_conversation_not_found(
    chat_service, mock_agent, mock_repository
):
    # Setup
    mock_conversation_id = uuid.uuid4()
    mock_repository.get.side_effect = models.ConversationNotFoundError()

    # Execute & Assert
    with pytest.raises(models.ConversationNotFoundError):
        await chat_service.execute_chat(
            MOCK_QUESTION, "Geni AI-3.5", mock_conversation_id
        )

    # Assert repository.get was called
    mock_repository.get.assert_called_once_with(mock_conversation_id)

    # Assert agent and save were NOT called (execution stopped at exception)
    mock_agent.execute_flow.assert_not_called()
    mock_repository.save.assert_not_called()


@pytest.mark.asyncio
async def test_adds_all_agent_responses(chat_service, mock_agent, mock_repository):
    # Setup
    mock_agent_responses = [
        models.AssistantMessage(
            content=MOCK_RESPONSE_1, usage=MOCK_USAGE, model_id="gpt-4"
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_2, usage=MOCK_USAGE, model_id="gpt-4"
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_3, usage=MOCK_USAGE, model_id="gpt-4"
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_4, usage=MOCK_USAGE, model_id="gpt-4"
        ),
    ]
    mock_conversation = models.Conversation(id=str(uuid.uuid4()))
    mock_agent.execute_flow.return_value = mock_agent_responses
    mock_repository.get.return_value = mock_conversation

    # Execute
    result = await chat_service.execute_chat(
        MOCK_QUESTION, "Geni AI-3.5", mock_conversation.id
    )

    # Assert agent called with question string and model name
    mock_agent.execute_flow.assert_called_once_with(
        question=MOCK_QUESTION, model_name="Geni AI-3.5"
    )

    # Assert all agent messages added in order
    assert len(result.messages) == 5  # 1 user + 4 agent
    assert result.messages[0].role == "user"
    assert result.messages[0].content == MOCK_QUESTION

    # Verify each agent message was added in correct order (use the mock data)
    for i, mock_response in enumerate(mock_agent_responses):
        actual_message = result.messages[i + 1]
        assert actual_message.role == mock_response.role
        assert actual_message.content == mock_response.content

    # Assert repository.save called with all messages
    mock_repository.save.assert_called_once()
    saved_conversation = mock_repository.save.call_args[0][0]
    assert len(saved_conversation.messages) == 5
    for msg in saved_conversation.messages[1:]:
        assert msg.role == "assistant"
