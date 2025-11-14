import fastapi.testclient
import pytest

from app.chat import dependencies
from app.entrypoints import fastapi as fastapi_app
from tests.fixtures import agent as agent_fixtures


@pytest.fixture
def client():
    fastapi_app.app.dependency_overrides[dependencies.get_chat_agent] = lambda: agent_fixtures.StubChatAgent()

    client = fastapi.testclient.TestClient(fastapi_app.app)

    yield client

    fastapi_app.app.dependency_overrides.clear()


def test_post_chat_nonexistent_conversation_returns_404(client):
    response = client.post(
        "/chat", json={"question": "Hello", "conversation_id": "nonexistent-id"}
    )

    assert response.status_code == 404


def test_post_chat_empty_question_returns_404(client):
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 404


def test_post_sync_chat_valid_question_returns_200(client):
    response = client.post("/chat", json={"question": "Hello, how are you?"})

    assert response.status_code == 200
    assert response.json()["messages"][0] == {
        "role": "user",
        "content": "Hello, how are you?",
    }
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "model": "geni-ai-3.5",
    }


def test_post_chat_with_existing_conversation_returns_200(client):
    response = client.post("/chat", json={"question": "Hello!"})
    assert response.status_code == 200

    conversation_id = response.json()["id"]

    response = client.post(
        "/chat",
        json={"question": "How's the weather?", "conversation_id": conversation_id},
    )
    assert response.status_code == 200

    assert response.json()["messages"][0] == {"role": "user", "content": "Hello!"}
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "model": "geni-ai-3.5",
    }
    assert response.json()["messages"][2] == {
        "role": "user",
        "content": "How's the weather?",
    }
    assert response.json()["messages"][3] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "model": "geni-ai-3.5",
    }
