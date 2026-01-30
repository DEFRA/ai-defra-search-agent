import uuid
from unittest.mock import AsyncMock, MagicMock

import fastapi.testclient
import pytest

from app.chat import models
from app.common import mongo
from app.entrypoints.api import app


@pytest.fixture
def client_override():
    """Create a TestClient with dependency_overrides suitable for unit tests."""

    def get_fresh_mongo_client():
        # Return a minimal object; repo implementations in tests are mocked anyway
        class DummyClient:
            def __init__(self):
                pass

            def get_database(self, _name):
                return None

        return DummyClient()

    def get_fresh_mongo_db():
        return get_fresh_mongo_client()

    app.dependency_overrides[mongo.get_db] = get_fresh_mongo_db
    app.dependency_overrides[mongo.get_mongo_client] = get_fresh_mongo_client

    yield fastapi.testclient.TestClient(app)

    app.dependency_overrides.clear()


def test_post_chat_queues_message_and_saves(client_override):
    test_client = client_override

    # Prepare mocks for dependencies
    mock_conversation_repository = AsyncMock()
    mock_sqs_client = AsyncMock()
    mock_sqs_client.__aenter__.return_value = mock_sqs_client
    mock_sqs_client.__aexit__.return_value = None

    mock_model_resolution_service = MagicMock()
    mock_model_resolution_service.resolve_model.return_value = MagicMock(
        name="Model", spec=["name"]
    )  # type: ignore
    mock_model_resolution_service.resolve_model.return_value.name = "MockModel"

    # Override dependencies on the app
    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_conversation_repository] = (
        lambda: mock_conversation_repository
    )
    app.dependency_overrides[dependencies.get_sqs_client] = lambda: mock_sqs_client
    app.dependency_overrides[dependencies.get_model_resolution_service] = (
        lambda: mock_model_resolution_service
    )

    body = {"question": "Hello", "modelId": "mid"}

    resp = test_client.post("/chat", json=body)

    assert resp.status_code == 202
    resp_json = resp.json()
    assert "message_id" in resp_json
    assert "conversation_id" in resp_json

    # Ensure repo.save was called synchronously
    mock_conversation_repository.save.assert_awaited()
    # SQS send happens in background task but executes before TestClient returns
    mock_sqs_client.send_message.assert_awaited()


def test_post_chat_with_nonexistent_conversation_returns_404(client_override):
    test_client = client_override

    mock_conversation_repository = AsyncMock()
    mock_conversation_repository.get.return_value = None
    mock_sqs_client = AsyncMock()
    mock_sqs_client.__aenter__.return_value = mock_sqs_client
    mock_sqs_client.__aexit__.return_value = None

    mock_model_resolution_service = MagicMock()
    mock_model_resolution_service.resolve_model.return_value = MagicMock(name="Model")

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_conversation_repository] = (
        lambda: mock_conversation_repository
    )
    app.dependency_overrides[dependencies.get_sqs_client] = lambda: mock_sqs_client
    app.dependency_overrides[dependencies.get_model_resolution_service] = (
        lambda: mock_model_resolution_service
    )

    body = {
        "question": "Hi",
        "modelId": "mid",
        "conversationId": str(uuid.uuid4()),
    }

    resp = test_client.post("/chat", json=body)
    assert resp.status_code == 404


def test_get_conversation_not_found(client_override):
    test_client = client_override

    mock_conversation_repository = AsyncMock()
    mock_conversation_repository.get.return_value = None

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_conversation_repository] = (
        lambda: mock_conversation_repository
    )

    resp = test_client.get(f"/conversations/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_conversation_returns_data(client_override):
    test_client = client_override

    # Build a conversation with messages
    conversation = models.Conversation()
    user_msg = models.UserMessage(content="q", model_id="m", model_name="mn")
    assistant_msg = models.AssistantMessage(
        content="a",
        model_id="m",
        model_name="mn",
        usage=models.TokenUsage(1, 2, 3),
    )
    conversation.add_message(user_msg)
    conversation.add_message(assistant_msg)

    mock_conversation_repository = AsyncMock()
    mock_conversation_repository.get.return_value = conversation

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_conversation_repository] = (
        lambda: mock_conversation_repository
    )

    resp = test_client.get(f"/conversations/{conversation.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == str(conversation.id)
    assert len(data["messages"]) == 2
