import dataclasses
import uuid

import pytest

from app.chat import agent, models, repository, service
from app.config import BedrockModelConfig

MOCK_QUESTION = "What is the question?"
MOCK_RESPONSE_1 = "First response"
MOCK_RESPONSE_2 = "Second response"
MOCK_RESPONSE_3 = "Third response"
MOCK_RESPONSE_4 = "Fourth response"
MOCK_PRIOR_MESSAGE = "Previous question"
MOCK_MODEL_NAME = "GPT 4"
MOCK_MODEL_ID = "gpt-4"
MOCK_USAGE = models.TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20)


@pytest.fixture
def mock_agent(mocker):
    """Async mock of AbstractChatAgent"""
    return mocker.AsyncMock(spec=agent.AbstractChatAgent)


@pytest.fixture
def mock_repository(mocker):
    """Async mock of AbstractConversationRepository"""
    return mocker.AsyncMock(spec=repository.AbstractConversationRepository)


@pytest.fixture
def mock_model_resolution_service(mocker):
    resolver = mocker.Mock()
    resolver.resolve_model = mocker.Mock(
        return_value=BedrockModelConfig(
            name=MOCK_MODEL_NAME,
            bedrock_model_id="gpt-4-bedrock-id",
            model_id=MOCK_MODEL_ID,
            description="A conversational AI model optimized for dialogue.",
            guardrails=None,
        )
    )

    return resolver


@pytest.fixture
def chat_service(mock_agent, mock_repository, mock_model_resolution_service, mocker):
    """ChatService instance with mocked dependencies"""
    return service.ChatService(
        chat_agent=mock_agent,
        conversation_repository=mock_repository,
        model_resolution_service=mock_model_resolution_service,
        sqs_client=mocker.MagicMock(),
    )


@pytest.fixture
def mock_existing_conversation():
    """Conversation with prior messages"""
    prior_messages = [
        models.UserMessage(
            content=MOCK_PRIOR_MESSAGE, model_id="gpt-4", model_name="GPT-4"
        )
    ]
    return models.Conversation(id=str(uuid.uuid4()), messages=prior_messages.copy())


@pytest.mark.asyncio
async def test_executes_with_existing_conversation(
    chat_service, mock_agent, mock_repository, mock_existing_conversation
):
    # Setup
    mock_agent_responses = [
        models.AssistantMessage(
            content=MOCK_RESPONSE_1,
            usage=MOCK_USAGE,
            model_id="gpt-4",
            model_name="GPT-4",
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_2,
            usage=MOCK_USAGE,
            model_id="gpt-4",
            model_name="GPT-4",
        ),
    ]
    mock_agent.execute_flow.return_value = mock_agent_responses
    mock_repository.get.return_value = mock_existing_conversation

    result = await chat_service.execute_chat(
        MOCK_QUESTION,
        "Geni AI-3.5",
        uuid.uuid4(),
        mock_existing_conversation.id,
    )

    # Assert repository.get called correctly
    mock_repository.get.assert_called_once_with(mock_existing_conversation.id)

    # Assert agent called with question, model name, and conversation history
    mock_agent.execute_flow.assert_called_once()
    call_args = mock_agent.execute_flow.call_args[0]
    agent_request = call_args[0]
    assert isinstance(agent_request, models.AgentRequest)
    assert agent_request.question == MOCK_QUESTION
    assert agent_request.model_id == "Geni AI-3.5"
    # History should include the prior message only (not the new user message)
    assert len(agent_request.conversation) == 1
    assert agent_request.conversation[0].content == MOCK_PRIOR_MESSAGE

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
            content=MOCK_RESPONSE_1,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
    ]
    mock_agent.execute_flow.return_value = mock_agent_responses

    result = await chat_service.execute_chat(MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4())

    # Assert repository.get NOT called
    mock_repository.get.assert_not_called()

    # Assert agent called with question, model name, and empty conversation history
    mock_agent.execute_flow.assert_called_once()
    call_args = mock_agent.execute_flow.call_args[0]
    agent_request = call_args[0]
    assert isinstance(agent_request, models.AgentRequest)
    assert agent_request.question == MOCK_QUESTION
    assert agent_request.model_id == MOCK_MODEL_ID
    # For new conversation, history should be empty list (no prior messages)
    assert agent_request.conversation == []

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

    with pytest.raises(models.ConversationNotFoundError):
        await chat_service.execute_chat(
            MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4(), mock_conversation_id
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
            content=MOCK_RESPONSE_1,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_2,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_3,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
        models.AssistantMessage(
            content=MOCK_RESPONSE_4,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
    ]
    mock_conversation = models.Conversation(id=str(uuid.uuid4()))
    mock_agent.execute_flow.return_value = mock_agent_responses
    mock_repository.get.return_value = mock_conversation

    result = await chat_service.execute_chat(
        MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4(), mock_conversation.id
    )

    # Assert agent called with question, model name, and conversation history
    mock_agent.execute_flow.assert_called_once()
    call_args = mock_agent.execute_flow.call_args[0]
    agent_request = call_args[0]
    assert isinstance(agent_request, models.AgentRequest)
    assert agent_request.question == MOCK_QUESTION
    assert agent_request.model_id == MOCK_MODEL_ID
    # For empty conversation, history should be empty list
    assert agent_request.conversation == []

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


@pytest.mark.asyncio
async def test_execute_chat_with_multi_turn_conversation_includes_full_history(
    chat_service, mock_agent, mock_repository
):
    """Test that all previous messages are included in conversation history"""
    # Setup - conversation with multiple turns
    conversation_id = uuid.uuid4()
    existing_conversation = models.Conversation(
        id=conversation_id,
        messages=[
            models.UserMessage(
                content="What is Python?",
                model_id=MOCK_MODEL_ID,
                model_name=MOCK_MODEL_NAME,
            ),
            models.AssistantMessage(
                content="Python is a programming language.",
                model_id=MOCK_MODEL_ID,
                model_name=MOCK_MODEL_NAME,
                usage=MOCK_USAGE,
            ),
            models.UserMessage(
                content="Who created it?",
                model_id=MOCK_MODEL_ID,
                model_name=MOCK_MODEL_NAME,
            ),
            models.AssistantMessage(
                content="It was created by Guido van Rossum.",
                model_id=MOCK_MODEL_ID,
                model_name=MOCK_MODEL_NAME,
                usage=MOCK_USAGE,
            ),
        ],
    )
    mock_repository.get.return_value = existing_conversation
    mock_agent_responses = [
        models.AssistantMessage(
            content="Python was first released in 1991.",
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        ),
    ]
    mock_agent.execute_flow.return_value = mock_agent_responses

    result = await chat_service.execute_chat(
        question="When was it created?",
        model_id=MOCK_MODEL_ID,
        message_id=uuid.uuid4(),
        conversation_id=conversation_id,
    )

    # Assert the agent was called with all 4 previous messages as history
    mock_agent.execute_flow.assert_called_once()
    call_args = mock_agent.execute_flow.call_args[0]
    agent_request = call_args[0]
    assert isinstance(agent_request, models.AgentRequest)
    assert agent_request.question == "When was it created?"
    assert agent_request.model_id == MOCK_MODEL_ID

    # History should have all 4 previous messages (not including the new user message we just added)
    history = agent_request.conversation
    assert len(history) == 4
    assert history[0].content == "What is Python?"
    assert history[0].role == "user"
    assert history[1].content == "Python is a programming language."
    assert history[1].role == "assistant"
    assert history[2].content == "Who created it?"
    assert history[2].role == "user"
    assert history[3].content == "It was created by Guido van Rossum."
    assert history[3].role == "assistant"

    # Assert final conversation has all 6 messages
    assert len(result.messages) == 6
    assert result.messages[4].content == "When was it created?"
    assert result.messages[4].role == "user"
    assert result.messages[5].content == "Python was first released in 1991."
    assert result.messages[5].role == "assistant"

    # Assert repository.save called
    mock_repository.save.assert_called_once()
    saved_conversation = mock_repository.save.call_args[0][0]
    assert len(saved_conversation.messages) == 6


@pytest.mark.asyncio
async def test_execute_chat_appends_rag_error_to_content(
    chat_service, mock_agent, mock_repository
):
    rag_error_msg = "RAG lookup failed. Knowledge base sources could not be retrieved."
    mock_agent.execute_flow.return_value = [
        models.AssistantMessage(
            content="Here is my response.",
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
            rag_error=rag_error_msg,
        )
    ]

    await chat_service.execute_chat(MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4())

    saved_conversation = mock_repository.save.call_args[0][0]
    assistant_msg = saved_conversation.messages[-1]
    assert assistant_msg.content == f"Here is my response.\n\n*{rag_error_msg}*"


@pytest.mark.asyncio
async def test_execute_chat_appends_sources_to_content(
    chat_service, mock_agent, mock_repository
):
    sources = [
        models.Source(
            name="Doc 1", location="http://doc1.com", snippet="Snippet 1", score=0.95
        ),
    ]
    mock_agent.execute_flow.return_value = [
        models.AssistantMessage(
            content="Answer with sources.",
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
            sources=sources,
        )
    ]

    await chat_service.execute_chat(MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4())

    saved_conversation = mock_repository.save.call_args[0][0]
    assistant_msg = saved_conversation.messages[-1]
    assert "Answer with sources." in assistant_msg.content
    assert "### Sources" in assistant_msg.content
    assert "Doc 1" in assistant_msg.content
    assert "http://doc1.com" in assistant_msg.content
    assert "Snippet 1" in assistant_msg.content


@pytest.mark.asyncio
async def test_execute_chat_appends_sources_and_rag_error_to_content(
    chat_service, mock_agent, mock_repository
):
    sources = [
        models.Source(
            name="Doc 1", location="http://doc1.com", snippet="Snippet 1", score=0.95
        ),
    ]
    rag_error_msg = "RAG lookup failed. Knowledge base sources could not be retrieved."
    mock_agent.execute_flow.return_value = [
        models.AssistantMessage(
            content="Partial answer.",
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
            sources=sources,
            rag_error=rag_error_msg,
        )
    ]

    await chat_service.execute_chat(MOCK_QUESTION, MOCK_MODEL_ID, uuid.uuid4())

    saved_conversation = mock_repository.save.call_args[0][0]
    assistant_msg = saved_conversation.messages[-1]
    assert "Partial answer." in assistant_msg.content
    assert "### Sources" in assistant_msg.content
    assert "Doc 1" in assistant_msg.content
    assert f"*{rag_error_msg}*" in assistant_msg.content


def test_build_knowledge_reference_str_formats_correctly(chat_service):
    sources = [
        models.Source(
            name="Doc 1", location="http://doc1.com", snippet="Snippet 1", score=0.95
        ),
        models.Source(
            name="Doc 2",
            location="http://doc2.com",
            snippet="Line 1\nLine 2",
            score=0.8,
        ),
    ]

    expected_output = (
        "\n\n### Sources\n\n"
        "1. **[Doc 1](http://doc1.com)** (95%)\n"
        "   > Snippet 1\n\n"
        "2. **[Doc 2](http://doc2.com)** (80%)\n"
        "   > Line 1\n"
        "   > Line 2"
    )

    result = chat_service._build_knowledge_reference_str(sources)
    assert result == expected_output


@pytest.mark.asyncio
async def test_execute_chat_does_not_duplicate_user_message(
    chat_service, mock_agent, mock_repository
):
    convo = models.Conversation(id=str(uuid.uuid4()))
    existing_message_id = uuid.uuid4()
    last_user = models.UserMessage(
        message_id=existing_message_id,
        content=MOCK_QUESTION,
        model_id=MOCK_MODEL_ID,
        model_name=MOCK_MODEL_NAME,
    )
    last_user = dataclasses.replace(last_user, status=models.MessageStatus.QUEUED)
    convo.messages.append(last_user)

    mock_repository.get.return_value = convo

    mock_agent.execute_flow.return_value = [
        models.AssistantMessage(
            content=MOCK_RESPONSE_1,
            usage=MOCK_USAGE,
            model_name=MOCK_MODEL_NAME,
            model_id=MOCK_MODEL_ID,
        )
    ]

    result = await chat_service.execute_chat(
        MOCK_QUESTION, MOCK_MODEL_ID, existing_message_id, convo.id
    )

    user_messages = [m for m in result.messages if m.role == "user"]
    assert len(user_messages) == 1
