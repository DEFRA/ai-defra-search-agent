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
def mock_conversation_repository():
    """Create a mock conversation repository."""
    return AsyncMock()


@pytest.fixture
def mock_sqs_client():
    """Create a mock SQS client with async context manager support."""
    mock = AsyncMock()
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    return mock


@pytest.fixture
def mock_model_resolution_service():
    """Create a mock model resolution service."""
    mock = MagicMock()
    mock.resolve_model.return_value = None  # Indicates valid model
    return mock


@pytest.fixture
def client(
    monkeypatch,
    mongo_uri,
    mock_conversation_repository,
    mock_sqs_client,
    mock_model_resolution_service,
):
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
    app.dependency_overrides[dependencies.get_conversation_repository] = (
        lambda: mock_conversation_repository
    )
    app.dependency_overrides[dependencies.get_sqs_client] = lambda: mock_sqs_client
    app.dependency_overrides[dependencies.get_model_resolution_service] = (
        lambda: mock_model_resolution_service
    )

    # Mock worker_task in app state for health checks
    mock_task = MagicMock()
    mock_task.done.return_value = False
    app.state.worker_task = mock_task

    test_client = fastapi.testclient.TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_post_chat_valid_question_returns_202(
    client, mock_conversation_repository, mock_sqs_client
):
    """Test POST /chat returns 202 with queued message details."""
    body = {"question": "Hello, how are you?", "modelId": "anthropic.claude-3-haiku"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED
    response_json = response.json()
    assert response_json["status"] == "queued"
    assert "message_id" in response_json
    assert "conversation_id" in response_json
    assert uuid.UUID(response_json["message_id"])  # Verify it's a valid UUID
    assert uuid.UUID(response_json["conversation_id"])  # Verify it's a valid UUID

    # Verify conversation was saved with message
    mock_conversation_repository.save.assert_called_once()
    saved_conversation = mock_conversation_repository.save.call_args[0][0]
    assert isinstance(saved_conversation, models.Conversation)
    assert len(saved_conversation.messages) == 1
    assert saved_conversation.messages[0].content == "Hello, how are you?"
    assert saved_conversation.messages[0].status == models.MessageStatus.QUEUED

    # Verify message was sent to SQS
    mock_sqs_client.send_message.assert_called_once()


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


def test_post_chat_unsupported_model_returns_400(client, mock_model_resolution_service):
    """Test POST /chat with unsupported model returns 400."""
    from app.models import UnsupportedModelError

    # Make resolve_model raise for unsupported model
    mock_model_resolution_service.resolve_model.side_effect = UnsupportedModelError(
        "Model invalid-model is not supported"
    )

    body = {"question": "Hello", "modelId": "invalid-model"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    mock_model_resolution_service.resolve_model.assert_called_once_with("invalid-model")


def test_post_chat_with_conversation_id(
    client, mock_conversation_repository, mock_sqs_client
):
    """Test POST /chat with existing conversation_id."""
    conversation_id = str(uuid.uuid4())
    body = {
        "question": "Follow-up question",
        "conversationId": conversation_id,
        "modelId": "anthropic.claude-3-haiku",
    }

    # Mock existing conversation
    existing_conversation = models.Conversation(
        id=uuid.UUID(conversation_id),
        messages=[
            models.UserMessage(
                content="Previous question",
                model_id="anthropic.claude-3-haiku",
                model_name="Claude 3 Haiku",
                message_id=uuid.uuid4(),
                status=models.MessageStatus.COMPLETED,
            )
        ],
    )
    mock_conversation_repository.get.return_value = existing_conversation

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED

    # Verify conversation was loaded
    mock_conversation_repository.get.assert_called_once_with(uuid.UUID(conversation_id))

    # Verify conversation was saved with new message
    saved_conversation = mock_conversation_repository.save.call_args[0][0]
    assert str(saved_conversation.id) == conversation_id
    assert len(saved_conversation.messages) == 2  # Previous + new message

    # Verify SQS message includes conversation_id
    sqs_message = mock_sqs_client.send_message.call_args[0][0]
    assert sqs_message["conversation_id"] == conversation_id


def test_get_conversation_by_id_returns_conversation(
    client, mock_conversation_repository
):
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
    mock_conversation_repository.get.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["conversation_id"] == str(conversation_id)
    assert len(response_json["messages"]) == 2

    mock_conversation_repository.get.assert_called_once_with(conversation_id)


def test_get_conversation_not_found_returns_404(client, mock_conversation_repository):
    """Test GET /conversations/{conversation_id} returns 404 when conversation doesn't exist."""
    conversation_id = uuid.uuid4()
    mock_conversation_repository.get.return_value = None

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_get_conversation_with_message_statuses(client, mock_conversation_repository):
    """Test GET /conversations/{conversation_id} returns messages with status."""
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
    mock_conversation_repository.get.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert len(response_json["messages"]) == 1
    assert response_json["messages"][0]["status"] == "processing"
    assert response_json["messages"][0]["message_id"] == str(message_id)


def test_get_conversation_with_failed_message(client, mock_conversation_repository):
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
                error_code=429,
            ),
        ],
    )
    mock_conversation_repository.get.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    message = response_json["messages"][0]
    assert message["status"] == "failed"
    assert message["error_message"] == "ThrottlingException: Rate limit exceeded"
    assert message["error_code"] == 429


def test_get_conversation_with_completed_messages(client, mock_conversation_repository):
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
    mock_conversation_repository.get.return_value = mock_conversation

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["conversation_id"] == str(conversation_id)
    assert len(response_json["messages"]) == 2
    assert response_json["messages"][0]["role"] == "user"
    assert response_json["messages"][0]["status"] == "completed"
    assert response_json["messages"][1]["role"] == "assistant"
