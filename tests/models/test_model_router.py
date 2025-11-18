import fastapi.testclient
import pytest

from app.entrypoints import fastapi as fastapi_app
from app.models import dependencies as model_dependencies


@pytest.fixture
def client():
    return fastapi.testclient.TestClient(fastapi_app.app)


def test_get_models_returns_model_list(client):
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert data == [
        {
            "modelName": "Geni AI 3.5",
            "modelDescription": "A powerful conversational AI model suitable for a wide range of tasks.",
        },
        {
            "modelName": "Geni AI 4",
            "modelDescription": "An advanced conversational AI model with enhanced understanding and generation capabilities.",
        },
    ]


def test_get_models_returns_204_when_no_models_configured(client):
    # Create a mock service that returns no models
    class EmptyModelResolutionService:
        def get_available_models(self):
            return []

    # Override the dependency for this test
    fastapi_app.app.dependency_overrides[
        model_dependencies.get_model_resolution_service
    ] = lambda: EmptyModelResolutionService()

    try:
        response = client.get("/models")
        assert response.status_code == 204
    finally:
        # Clean up the override after the test
        fastapi_app.app.dependency_overrides.clear()
