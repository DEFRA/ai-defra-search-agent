import re

import fastapi.testclient
import pymongo
import pytest

from app import config
from app.chat import dependencies
from app.common import mongo
from app.entrypoints.api import app


@pytest.fixture
def client(bedrock_inference_service):
    def get_fresh_mongo_client():
        match = re.search(
            r"mongodb://(?:[^@]+@)?([^:/]+)", config.get_config().mongo.uri
        )
        host = match.group(1) if match else "localhost"
        return pymongo.AsyncMongoClient(
            host, uuidRepresentation="standard", timeoutMS=5000
        )

    def get_fresh_mongo_db():
        client = get_fresh_mongo_client()
        return client.get_database("ai_defra_search_agent")

    app.dependency_overrides[mongo.get_db] = get_fresh_mongo_db
    app.dependency_overrides[mongo.get_mongo_client] = get_fresh_mongo_client

    app.dependency_overrides[dependencies.get_bedrock_inference_service] = (
        lambda: bedrock_inference_service
    )

    test_client = fastapi.testclient.TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_post_chat_nonexistent_conversation_returns_404(client):
    body = {
        "question": "Hello",
        "conversationId": "2c29818a-4367-4114-a789-4494a527b8af",
        "modelId": "geni-ai-3.5",
    }

    response = client.post("/chat", json=body)

    assert response.status_code == 404


def test_post_chat_empty_question_returns_400(client):
    body = {"question": "", "modelId": "geni-ai-3.5"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400


def test_post_chat_missing_model_name_returns_400(client):
    body = {"question": "Hello"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400


def test_post_chat_unsupported_model_returns_400(client):
    body = {"question": "Hello", "modelId": "unsupported-model-id"}

    response = client.post("/chat", json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == "Model 'unsupported-model-id' not found"


def test_post_sync_chat_valid_question_returns_200(client):
    body = {"question": "Hello, how are you?", "modelId": "geni-ai-3.5"}

    response = client.post("/chat", json=body)

    assert response.status_code == 200

    assert response.json()["conversationId"] is not None
    assert response.json()["messages"][0] == {
        "role": "user",
        "content": "Hello, how are you?",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }


def test_post_chat_with_existing_conversation_returns_200(client):
    start_body = {"question": "Hello!", "modelId": "geni-ai-3.5"}

    response = client.post("/chat", json=start_body)
    assert response.status_code == 200

    conversation_id = response.json()["conversationId"]

    continue_body = {
        "question": "How's the weather?",
        "conversationId": conversation_id,
        "modelId": "geni-ai-3.5",
    }

    response = client.post("/chat", json=continue_body)

    assert response.status_code == 200

    assert response.json()["conversationId"] is not None
    assert response.json()["messages"][0] == {
        "role": "user",
        "content": "Hello!",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }
    assert response.json()["messages"][1] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }
    assert response.json()["messages"][2] == {
        "role": "user",
        "content": "How's the weather?",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }
    assert response.json()["messages"][3] == {
        "role": "assistant",
        "content": "This is a stub response.",
        "modelId": "geni-ai-3.5",
        "modelName": "Geni AI 3.5",
    }
