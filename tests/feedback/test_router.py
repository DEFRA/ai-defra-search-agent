import uuid

import fastapi.testclient
import pymongo
import pytest

from app import config
from app.chat import dependencies
from app.common import mongo
from app.common.mongo import MongoUnavailableError
from app.entrypoints.api import app


@pytest.fixture
def client(monkeypatch, bedrock_inference_service, mongo_uri):
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

    app.dependency_overrides[dependencies.get_bedrock_inference_service] = (
        lambda: bedrock_inference_service
    )

    test_client = fastapi.testclient.TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_post_feedback_with_all_fields_returns_201(client):
    conversation_id = str(uuid.uuid4())
    body = {
        "conversationId": conversation_id,
        "wasHelpful": "very-useful",
        "comment": "This was very helpful!",
    }

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()
    assert response.json()["feedbackId"] is not None


def test_post_feedback_minimal_returns_201(client):
    body = {"wasHelpful": "not-useful"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()


def test_post_feedback_without_conversation_id_returns_201(client):
    body = {"wasHelpful": "useful", "comment": "Great response!"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()


def test_post_feedback_missing_required_field_returns_400(client):
    body = {"conversationId": str(uuid.uuid4()), "comment": "Missing wasHelpful"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400


def test_post_feedback_invalid_uuid_returns_400(client):
    body = {"conversationId": "not-a-uuid", "wasHelpful": "neither"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400


def test_post_feedback_comment_too_long_returns_400(client):
    long_comment = "x" * 1201
    body = {"wasHelpful": "not-at-all-useful", "comment": long_comment}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400


def test_post_feedback_with_neither_returns_201(client):
    """Test the 'neither' satisfaction value"""
    body = {"wasHelpful": "neither", "comment": "It was okay"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()


def test_post_feedback_with_not_at_all_useful_returns_201(client):
    """Test the 'not-at-all-useful' satisfaction value"""
    body = {"wasHelpful": "not-at-all-useful"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()


def test_post_feedback_with_empty_comment_returns_201(client):
    """Test that empty comment is allowed"""
    body = {"wasHelpful": "useful", "comment": ""}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()


def test_post_feedback_mongo_unavailable_returns_503(mocker, monkeypatch, mongo_uri):
    monkeypatch.setenv("MONGO_URI", mongo_uri)

    mock_feedback_service = mocker.AsyncMock()
    mock_feedback_service.submit_feedback.side_effect = MongoUnavailableError(
        "Service unavailable"
    )

    from app.feedback import dependencies as feedback_deps

    app.dependency_overrides[mongo.get_db] = lambda: None
    app.dependency_overrides[mongo.get_mongo_client] = lambda: None
    app.dependency_overrides[feedback_deps.get_feedback_service] = (
        lambda: mock_feedback_service
    )

    test_client = fastapi.testclient.TestClient(app)
    body = {"wasHelpful": "useful"}

    response = test_client.post("/feedback", json=body)

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Service unavailable" in response.json()["detail"]
