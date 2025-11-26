import re

import fastapi.testclient
import pymongo
import pytest

from app.chat import dependencies
from app.common import mongo
from app.entrypoints import fastapi as fastapi_app
from tests.fixtures import bedrock as bedrock_fixture


@pytest.fixture
def client():
    def get_fresh_mongo_client():
        match = re.search(
            r"mongodb://(?:[^@]+@)?([^:/]+)", fastapi_app.app_config.mongo.uri
        )
        host = match.group(1) if match else "localhost"
        return pymongo.AsyncMongoClient(
            host, uuidRepresentation="standard", timeoutMS=5000
        )

    def get_fresh_mongo_db():
        client = get_fresh_mongo_client()
        return client.get_database("ai_defra_search_agent")

    fastapi_app.app.dependency_overrides[mongo.get_db] = get_fresh_mongo_db
    fastapi_app.app.dependency_overrides[mongo.get_mongo_client] = (
        get_fresh_mongo_client
    )

    fastapi_app.app.dependency_overrides[dependencies.get_bedrock_inference_service] = (
        lambda: bedrock_fixture.StubBedrockInferenceService()
    )

    test_client = fastapi.testclient.TestClient(fastapi_app.app)

    yield test_client

    # Clean up
    fastapi_app.app.dependency_overrides.clear()


def test_post_chat_nonexistent_conversation_returns_404(client):
    body = {
        "question": "Hello",
        "conversationId": "2c29818a-4367-4114-a789-4494a527b8af",
        "modelName": "Geni AI 3.5",
    }

    response = client.post("/chat", json=body)

    assert response.status_code == 404


def test_post_chat_empty_question_returns_400(client):
    body = {"question": "", "modelName": "Geni AI 3.5"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400


def test_post_chat_missing_model_name_returns_400(client):
    body = {"question": "Hello"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400


def test_post_chat_nonsupported_model_returns_400(client):
    body = {"question": "Hello", "modelName": "Nonexistent Model"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400


def test_post_sync_chat_valid_question_returns_200(client):
    body = {"question": "Hello, how are you?", "modelName": "Geni AI 3.5"}

    response = client.post("/chat", json=body)

    assert response.status_code == 200

    assert response.json()["conversationId"] is not None
    assert response.json()["messages"][0] == {
        "role": "user",
        "content": "Hello, how are you?",
    }
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
    }


def test_post_chat_with_existing_conversation_returns_200(client):
    start_body = {"question": "Hello!", "modelName": "Geni AI 3.5"}

    response = client.post("/chat", json=start_body)
    assert response.status_code == 200

    conversation_id = response.json()["conversationId"]

    continue_body = {
        "question": "How's the weather?",
        "conversationId": conversation_id,
        "modelName": "Geni AI 3.5",
    }

    response = client.post("/chat", json=continue_body)

    assert response.status_code == 200

    assert response.json()["conversationId"] is not None
    assert response.json()["messages"][0] == {"role": "user", "content": "Hello!"}
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
    }
    assert response.json()["messages"][2] == {
        "role": "user",
        "content": "How's the weather?",
    }
    assert response.json()["messages"][3] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
    }
