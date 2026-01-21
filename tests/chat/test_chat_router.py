import uuid
from unittest.mock import AsyncMock, MagicMock

import fastapi.testclient
import pymongo
import pytest
from fastapi import status

from app import config
from app.chat import dependencies, job_models
from app.common import mongo
from app.entrypoints.api import app


@pytest.fixture
def mock_job_repository():
    """Create a mock job repository."""
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
def client(monkeypatch, mongo_uri, mock_job_repository, mock_sqs_client, mock_model_resolution_service):
    monkeypatch.setenv("MONGO_URI", mongo_uri)

    def get_fresh_mongo_client():
        return pymongo.AsyncMongoClient(
            config.get_config().mongo.uri, uuidRepresentation="standard", timeoutMS=5000
        )

    def get_fresh_mongo_db():
        client = get_fresh_mongo_client()
        return client.get_database("ai_defra_search_agent")

    app.dependency_overrides[mongo.get_db] = get_fresh_mongo_db
    app.dependency_overrides[mongo.get_mongo_client] = get_fresh_mongo_client
    app.dependency_overrides[dependencies.get_job_repository] = lambda: mock_job_repository
    app.dependency_overrides[dependencies.get_sqs_client] = lambda: mock_sqs_client
    app.dependency_overrides[dependencies.get_model_resolution_service] = lambda: mock_model_resolution_service

    test_client = fastapi.testclient.TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_post_chat_valid_question_returns_202(client, mock_job_repository, mock_sqs_client):
    """Test POST /chat returns 202 with job_id and queued status."""
    body = {"question": "Hello, how are you?", "modelId": "anthropic.claude-3-haiku"}

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED
    response_json = response.json()
    assert "job_id" in response_json
    assert response_json["status"] == "queued"
    assert uuid.UUID(response_json["job_id"])  # Verify it's a valid UUID

    # Verify job was created in repository
    mock_job_repository.create.assert_called_once()
    created_job = mock_job_repository.create.call_args[0][0]
    assert isinstance(created_job, job_models.ChatJob)
    assert created_job.question == "Hello, how are you?"
    assert created_job.model_id == "anthropic.claude-3-haiku"

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


def test_post_chat_with_conversation_id(client, mock_job_repository, mock_sqs_client):
    """Test POST /chat with existing conversation_id."""
    conversation_id = str(uuid.uuid4())
    body = {
        "question": "Follow-up question",
        "conversationId": conversation_id,
        "modelId": "anthropic.claude-3-haiku"
    }

    response = client.post("/chat", json=body)

    assert response.status_code == status.HTTP_202_ACCEPTED

    # Verify job was created with conversation_id
    created_job = mock_job_repository.create.call_args[0][0]
    assert str(created_job.conversation_id) == conversation_id

    # Verify SQS message includes conversation_id
    sqs_message = mock_sqs_client.send_message.call_args[0][0]
    assert sqs_message["conversation_id"] == conversation_id


def test_get_job_by_id_returns_job(client, mock_job_repository):
    """Test GET /jobs/{job_id} returns job details."""
    job_id = uuid.uuid4()
    mock_job = job_models.ChatJob(
        job_id=job_id,
        question="Test question",
        model_id="anthropic.claude-3-haiku",
        status=job_models.JobStatus.COMPLETED,
        result={"conversation_id": str(uuid.uuid4()), "messages": []}
    )
    mock_job_repository.get.return_value = mock_job

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["job_id"] == str(job_id)
    assert response_json["status"] == "completed"
    assert "result" in response_json

    mock_job_repository.get.assert_called_once_with(job_id)


def test_get_job_not_found_returns_404(client, mock_job_repository):
    """Test GET /jobs/{job_id} returns 404 when job doesn't exist."""
    job_id = uuid.uuid4()
    mock_job_repository.get.return_value = None

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_get_job_completed_with_result(client, mock_job_repository):
    """Test GET /jobs/{job_id} returns completed job with full result."""
    job_id = uuid.uuid4()
    conversation_id = str(uuid.uuid4())
    result = {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": "user",
                "content": "What is AI?",
                "model_name": "Claude Sonnet 3.7",
                "model_id": "anthropic.claude-3-haiku",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "role": "assistant",
                "content": "AI is artificial intelligence",
                "model_name": "Claude Sonnet 3.7",
                "model_id": "anthropic.claude-3-haiku",
                "timestamp": "2024-01-01T00:00:01"
            }
        ]
    }

    mock_job = job_models.ChatJob(
        job_id=job_id,
        question="What is AI?",
        model_id="anthropic.claude-3-haiku",
        status=job_models.JobStatus.COMPLETED,
        result=result
    )
    mock_job_repository.get.return_value = mock_job

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["status"] == "completed"
    assert response_json["result"]["conversation_id"] == conversation_id
    assert len(response_json["result"]["messages"]) == 2


def test_get_job_failed_with_error_code(client, mock_job_repository):
    """Test GET /jobs/{job_id} returns failed job with error details."""
    job_id = uuid.uuid4()
    mock_job = job_models.ChatJob(
        job_id=job_id,
        question="Test question",
        model_id="anthropic.claude-3-haiku",
        status=job_models.JobStatus.FAILED,
        error_message="ThrottlingException: Rate limit exceeded",
        error_code=429
    )
    mock_job_repository.get.return_value = mock_job

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["status"] == "failed"
    assert response_json["error_message"] == "ThrottlingException: Rate limit exceeded"
    assert response_json["error_code"] == 429


def test_get_job_processing_status(client, mock_job_repository):
    """Test GET /jobs/{job_id} returns job in processing state."""
    job_id = uuid.uuid4()
    mock_job = job_models.ChatJob(
        job_id=job_id,
        question="Test question",
        model_id="anthropic.claude-3-haiku",
        status=job_models.JobStatus.PROCESSING
    )
    mock_job_repository.get.return_value = mock_job

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["status"] == "processing"
    assert response_json["result"] is None

