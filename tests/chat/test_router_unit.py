import uuid
from unittest.mock import AsyncMock

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

    mock_chat_service = AsyncMock()
    mock_chat_service.queue_chat.return_value = (
        uuid.uuid4(),
        uuid.uuid4(),
        models.MessageStatus.QUEUED,
    )

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    body = {"question": "Hello", "modelId": "mid"}

    resp = test_client.post("/chat", json=body)

    assert resp.status_code == 202
    resp_json = resp.json()
    assert "message_id" in resp_json
    assert "conversation_id" in resp_json
    assert resp_json["status"] == "queued"

    mock_chat_service.queue_chat.assert_awaited_once_with(
        question="Hello", model_id="mid", conversation_id=None
    )


def test_post_chat_with_nonexistent_conversation_returns_404(client_override):
    test_client = client_override

    mock_chat_service = AsyncMock()
    mock_chat_service.queue_chat.side_effect = models.ConversationNotFoundError(
        "Conversation not found"
    )

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    body = {
        "question": "Hi",
        "modelId": "mid",
        "conversationId": str(uuid.uuid4()),
    }

    resp = test_client.post("/chat", json=body)
    assert resp.status_code == 404


def test_get_conversation_not_found(client_override):
    test_client = client_override

    mock_chat_service = AsyncMock()
    mock_chat_service.get_conversation.return_value = None

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    resp = test_client.get(f"/conversations/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_conversation_returns_data(client_override):
    test_client = client_override

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

    mock_chat_service = AsyncMock()
    mock_chat_service.get_conversation.return_value = conversation

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    resp = test_client.get(f"/conversations/{conversation.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == str(conversation.id)
    assert len(data["messages"]) == 2


def test_post_chat_with_unsupported_model_returns_400(client_override):
    test_client = client_override

    mock_chat_service = AsyncMock()
    from app.models import UnsupportedModelError

    mock_chat_service.queue_chat.side_effect = UnsupportedModelError("bad model")

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    body = {"question": "Hi", "modelId": "invalid"}

    resp = test_client.post("/chat", json=body)
    assert resp.status_code == 400
    assert "bad model" in resp.text


def test_post_chat_with_existing_conversation(client_override):
    test_client = client_override

    conv_id = uuid.uuid4()
    msg_id = uuid.uuid4()
    mock_chat_service = AsyncMock()
    mock_chat_service.queue_chat.return_value = (
        msg_id,
        conv_id,
        models.MessageStatus.QUEUED,
    )

    from app.chat import dependencies

    app.dependency_overrides[dependencies.get_chat_service] = lambda: mock_chat_service

    body = {
        "question": "Follow up",
        "modelId": "mid",
        "conversationId": str(conv_id),
    }

    resp = test_client.post("/chat", json=body)
    assert resp.status_code == 202
    resp_json = resp.json()
    assert resp_json["conversation_id"] == str(conv_id)
    assert resp_json["message_id"] == str(msg_id)

    mock_chat_service.queue_chat.assert_awaited_once_with(
        question="Follow up", model_id="mid", conversation_id=conv_id
    )
