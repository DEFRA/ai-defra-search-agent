from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 404  # Root endpoint doesn't exist, should return 404


def test_example():
    response = client.get("/example/test")
    assert (
        response.status_code == 404
    )  # This endpoint doesn't exist, so should return 404


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
