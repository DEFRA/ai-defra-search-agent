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
        return pymongo.AsyncMongoClient(
            config.get_config().mongo.uri, uuidRepresentation="standard", timeoutMS=5000
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
    assert response.json()["messages"][0]["role"] == "user"
    assert response.json()["messages"][0]["content"] == "Hello, how are you?"
    assert response.json()["messages"][0]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][0]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][0]

    assert response.json()["messages"][1]["role"] == "assistant"
    assert response.json()["messages"][1]["content"] == "This is a stub response."
    assert response.json()["messages"][1]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][1]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][1]


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

    assert response.json()["messages"][0]["role"] == "user"
    assert response.json()["messages"][0]["content"] == "Hello!"
    assert response.json()["messages"][0]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][0]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][0]

    assert response.json()["messages"][1]["role"] == "assistant"
    assert response.json()["messages"][1]["content"] == "This is a stub response."
    assert response.json()["messages"][1]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][1]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][1]

    assert response.json()["messages"][2]["role"] == "user"
    assert response.json()["messages"][2]["content"] == "How's the weather?"
    assert response.json()["messages"][2]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][2]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][2]

    assert response.json()["messages"][3]["role"] == "assistant"
    assert response.json()["messages"][3]["content"] == "This is a stub response."
    assert response.json()["messages"][3]["modelId"] == "geni-ai-3.5"
    assert response.json()["messages"][3]["modelName"] == "Geni AI 3.5"
    assert "timestamp" in response.json()["messages"][3]


def test_post_chat_bedrock_400_error_returns_400(mocker, client):
    # Mock the bedrock service to raise an HTTPException (like it would when AWS returns 400)
    from app.bedrock import service as bedrock_service

    mock_bedrock_service = mocker.Mock(spec=bedrock_service.BedrockInferenceService)
    mock_bedrock_service.invoke_anthropic.side_effect = fastapi.HTTPException(
        status_code=400, detail="Invalid request to AI model: Invalid model parameters"
    )

    # Override the dependency to use our mock
    app.dependency_overrides[dependencies.get_bedrock_inference_service] = (
        lambda: mock_bedrock_service
    )

    try:
        body = {"question": "Hello, how are you?", "modelId": "geni-ai-3.5"}

        response = client.post("/chat", json=body)

        assert response.status_code == 400
        assert "Invalid request to AI model" in response.json()["detail"]
        assert "Invalid model parameters" in response.json()["detail"]
    finally:
        # Clean up the override
        app.dependency_overrides.clear()
