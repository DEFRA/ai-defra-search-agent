import uuid
from unittest.mock import AsyncMock, MagicMock

import fastapi.testclient
import pymongo
import pytest
from fastapi import status

from app import config
from app.chat import models
from app.common import mongo
from app.entrypoints.api import app


@pytest.fixture
def mock_chat_service():
    """Create a mock chat service."""
    return AsyncMock()


@pytest.fixture
def client(monkeypatch, mongo_uri, mock_chat_service):
    monkeypatch.setenv("MONGO_URI", mongo_uri)

    def get_fresh_mongo_client():
        return pymongo.AsyncMongoClient(
            config.get_config().mongo.uri, uuidRepresentation="standard", timeoutMS=5000
        )

    def get_fresh_mongo_db():
        client = get_fresh_mongo_client()
        return client.get_database("ai_defra_search_agent")

    from app.chat import dependencies

    app.dependency_overrides[mongo.get_db] = get_fresh_mongo_db
    app.dependency_overrides[mongo.get_mongo_client] = get_fresh_mongo_client
    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    mock_task = MagicMock()
    mock_task.done.return_value = False
    app.state.worker_task = mock_task

    test_client = fastapi.testclient.TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_post_chat_valid_question_returns_202(client, mock_chat_service):
    """Test POST /chat returns 202 with queued message details."""
    msg_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    mock_chat_service.queue_chat.return_value = (
        msg_id,
        conv_id,
        models.MessageStatus.QUEUED,
    )

    body = {"question": "Hello, how are you?", "modelId": "anthropic.claude-3-haiku"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED
    response_json = response.json()
    assert response_json["status"] == "queued"
    assert response_json["message_id"] == str(msg_id)
    assert response_json["conversation_id"] == str(conv_id)

    mock_chat_service.queue_chat.assert_awaited_once_with(
        question="Hello, how are you?",
        model_id="anthropic.claude-3-haiku",
        conversation_id=None,
    )


def test_post_chat_empty_question_returns_400(client):
    """Test POST /chat with empty question returns 400."""
    body = {"question": "", "modelId": "anthropic.claude-3-haiku"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_post_chat_missing_model_id_returns_400(client):
    """Test POST /chat without model_id returns 400."""
    body = {"question": "Hello"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_post_chat_unsupported_model_returns_400(client, mock_chat_service):
    """Test POST /chat with unsupported model returns 400."""
    from app.models import UnsupportedModelError

    mock_chat_service.queue_chat.side_effect = UnsupportedModelError(
        "Model invalid-model is not supported"
    )

    body = {"question": "Hello", "modelId": "invalid-model"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    mock_chat_service.queue_chat.assert_awaited_once_with(
        question="Hello", model_id="invalid-model", conversation_id=None
    )


def test_post_chat_with_conversation_id(client, mock_chat_service):
    """Test POST /chat with existing conversation_id."""
    conversation_id = uuid.uuid4()
    msg_id = uuid.uuid4()

    mock_chat_service.queue_chat.return_value = (
        msg_id,
        conversation_id,
        models.MessageStatus.QUEUED,
    )

    body = {
        "question": "Follow-up question",
        "conversationId": str(conversation_id),
        "modelId": "anthropic.claude-3-haiku",
    }

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED
    response_json = response.json()
    assert response_json["conversation_id"] == str(conversation_id)
    assert response_json["message_id"] == str(msg_id)

    mock_chat_service.queue_chat.assert_awaited_once_with(
        question="Follow-up question",
        model_id="anthropic.claude-3-haiku",
        conversation_id=conversation_id,
    )


def test_get_conversation_by_id_returns_conversation(client, mock_chat_service):
    """Test GET /conversations/{conversation_id} returns conversation details."""
    conversation_id = uuid.uuid4()
    mock_conversation = models.Conversation(
        id=conversation_id,
        messages=[
            models.UserMessage(
                content="Test question",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                message_id=uuid.uuid4(),
                status=models.MessageStatus.COMPLETED,
            ),
            models.AssistantMessage(
                content="Test answer",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                usage=models.TokenUsage(
                    input_tokens=10, output_tokens=20, total_tokens=30
                ),
            ),
        ],
    )
    mock_chat_service.get_conversation.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["conversation_id"] == str(conversation_id)
    assert len(response_json["messages"]) == 2

    mock_chat_service.get_conversation.assert_awaited_once_with(conversation_id)


def test_get_conversation_not_found_returns_404(client, mock_chat_service):
    """Test GET /conversations/{conversation_id} returns 404 when conversation doesn't exist."""
    conversation_id = uuid.uuid4()
    mock_chat_service.get_conversation.side_effect = models.ConversationNotFoundError(
        "Conversation not found"
    )

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_get_conversation_with_message_statuses(client, mock_chat_service):
    """Test GET /conversations/{conversation_id} returns message with processing status."""
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    mock_conversation = models.Conversation(
        id=conversation_id,
        messages=[
            models.UserMessage(
                content="What is AI?",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                message_id=message_id,
                status=models.MessageStatus.PROCESSING,
            ),
        ],
    )
    mock_chat_service.get_conversation.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert len(response_json["messages"]) == 1
    assert response_json["messages"][0]["status"] == "processing"
    assert response_json["messages"][0]["message_id"] == str(message_id)


def test_get_conversation_with_failed_message(client, mock_chat_service):
    """Test GET /conversations/{conversation_id} returns failed message with error."""
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    mock_conversation = models.Conversation(
        id=conversation_id,
        messages=[
            models.UserMessage(
                content="Test question",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                message_id=message_id,
                status=models.MessageStatus.FAILED,
                error_message="ThrottlingException: Rate limit exceeded",
            ),
        ],
    )
    mock_chat_service.get_conversation.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    message = response_json["messages"][0]
    assert message["status"] == "failed"
    assert message["error_message"] == "ThrottlingException: Rate limit exceeded"


def test_get_conversation_with_completed_messages(client, mock_chat_service):
    """Test GET /conversations/{conversation_id} returns conversation with completed exchange."""
    conversation_id = uuid.uuid4()
    mock_conversation = models.Conversation(
        id=conversation_id,
        messages=[
            models.UserMessage(
                content="What is AI?",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                message_id=uuid.uuid4(),
                status=models.MessageStatus.COMPLETED,
            ),
            models.AssistantMessage(
                content="AI is artificial intelligence",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                usage=models.TokenUsage(
                    input_tokens=10, output_tokens=30, total_tokens=40
                ),
            ),
        ],
    )
    mock_chat_service.get_conversation.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["conversation_id"] == str(conversation_id)
    assert len(response_json["messages"]) == 2
    assert response_json["messages"][0]["role"] == "user"
    assert response_json["messages"][0]["status"] == "completed"
    assert response_json["messages"][1]["role"] == "assistant"
