import re
import uuid

import fastapi.testclient
import pymongo
import pytest

from app.common import mongo
from app.entrypoints import fastapi as fastapi_app


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

    test_client = fastapi.testclient.TestClient(fastapi_app.app)

    yield test_client

    # Clean up
    fastapi_app.app.dependency_overrides.clear()


def test_post_feedback_with_all_fields_returns_201(client):
    conversation_id = str(uuid.uuid4())
    body = {
        "conversationId": conversation_id,
        "wasHelpful": True,
        "comment": "This was very helpful!",
    }

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()
    assert response.json()["feedbackId"] is not None


def test_post_feedback_minimal_returns_201(client):
    body = {"wasHelpful": False}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()


def test_post_feedback_without_conversation_id_returns_201(client):
    body = {"wasHelpful": True, "comment": "Great response!"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 201
    assert "feedbackId" in response.json()
    assert "timestamp" in response.json()


def test_post_feedback_missing_required_field_returns_400(client):
    body = {"conversationId": str(uuid.uuid4()), "comment": "Missing wasHelpful"}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400


def test_post_feedback_invalid_uuid_returns_400(client):
    body = {"conversationId": "not-a-uuid", "wasHelpful": True}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400


def test_post_feedback_comment_too_long_returns_400(client):
    long_comment = "x" * 1001
    body = {"wasHelpful": True, "comment": long_comment}

    response = client.post("/feedback", json=body)

    assert response.status_code == 400
