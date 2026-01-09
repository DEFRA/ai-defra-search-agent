import fastapi.testclient
import pymongo
import pytest
from botocore.exceptions import ClientError

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


def test_post_chat_bedrock_throttling_error_returns_429(bedrock_inference_service, mocker):
    """Test that Bedrock throttling errors return 429. Client uses conversation ID from their request to retrieve conversation."""
    # Mock the bedrock service to raise a ClientError
    mocker.patch.object(
        bedrock_inference_service,
        'invoke_anthropic',
        side_effect=ClientError(
            {
                'Error': {
                    'Code': 'ThrottlingException',
                    'Message': 'Rate limit exceeded'
                },
                'ResponseMetadata': {
                    'HTTPStatusCode': 429
                }
            },
            'converse'
        )
    )

    # Setup test client with mocked service
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

    body = {"question": "Hello", "modelId": "geni-ai-3.5"}
    response = test_client.post("/chat", json=body)

    # Verify standard HTTP error response
    assert response.status_code == 429
    response_json = response.json()
    # ClientError string representation includes both error code and message
    assert 'ThrottlingException' in response_json['detail']
    assert 'Rate limit exceeded' in response_json['detail']

    app.dependency_overrides.clear()


def test_post_chat_bedrock_validation_error_returns_400(bedrock_inference_service, mocker):
    """Test that Bedrock validation errors return 400."""
    mocker.patch.object(
        bedrock_inference_service,
        'invoke_anthropic',
        side_effect=ClientError(
            {
                'Error': {
                    'Code': 'ValidationException',
                    'Message': 'Invalid input parameters'
                },
                'ResponseMetadata': {
                    'HTTPStatusCode': 400
                }
            },
            'converse'
        )
    )

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

    body = {"question": "Hello", "modelId": "geni-ai-3.5"}
    response = test_client.post("/chat", json=body)

    # Verify standard HTTP error response
    assert response.status_code == 400
    response_json = response.json()
    assert 'ValidationException' in response_json['detail']
    assert 'Invalid input parameters' in response_json['detail']

    app.dependency_overrides.clear()


def test_post_chat_bedrock_internal_error_returns_500(bedrock_inference_service, mocker):
    """Test that Bedrock internal errors return 500."""
    mocker.patch.object(
        bedrock_inference_service,
        'invoke_anthropic',
        side_effect=ClientError(
            {
                'Error': {
                    'Code': 'InternalServerException',
                    'Message': 'Internal server error'
                },
                'ResponseMetadata': {
                    'HTTPStatusCode': 500
                }
            },
            'converse'
        )
    )

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

    body = {"question": "Hello", "modelId": "geni-ai-3.5"}
    response = test_client.post("/chat", json=body)

    # Verify standard HTTP error response
    assert response.status_code == 500
    response_json = response.json()
    assert 'InternalServerException' in response_json['detail']
    assert 'Internal server error' in response_json['detail']

    app.dependency_overrides.clear()