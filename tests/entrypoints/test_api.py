import pytest
from fastapi.testclient import TestClient

from app.common import mongo
from app.entrypoints.api import app


@pytest.fixture
def healthy_mongo_client(mocker):
    mock_client = mocker.AsyncMock()
    mock_client.admin.command = mocker.AsyncMock(return_value={"ok": 1})
    return mock_client


@pytest.fixture
def client_with_mongo(healthy_mongo_client):
    app.dependency_overrides[mongo.get_mongo_client] = lambda: healthy_mongo_client
    yield TestClient(app)
    app.dependency_overrides.pop(mongo.get_mongo_client, None)


def test_lifespan(mocker):
    mock_mongo_client = mocker.AsyncMock()
    mock_get_mongo = mocker.patch(
        "app.common.mongo.get_mongo_client", return_value=mock_mongo_client
    )

    # Mock the worker task
    mock_worker = mocker.patch("app.entrypoints.api.run_worker")
    mock_worker.return_value = mocker.AsyncMock()

    # Using TestClient as a context manager triggers lifespan startup/shutdown
    with TestClient(app):
        mock_get_mongo.assert_called_once()  # Startup: connect called

    mock_mongo_client.close.assert_awaited_once()  # Shutdown: close called


def test_health(mocker, client_with_mongo):
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = False
    mocker.patch.object(app.state, "worker_task", mock_task)

    response = client_with_mongo.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_worker_not_running(mocker, client_with_mongo):
    mocker.patch.object(app.state, "worker_task", None)

    response = client_with_mongo.get("/health")
    assert response.status_code == 503
    assert "Worker task not running" in response.json()["detail"]


def test_health_worker_crashed(mocker, client_with_mongo):
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = True
    mock_task.result.side_effect = Exception("Worker crashed")
    mocker.patch.object(app.state, "worker_task", mock_task)

    response = client_with_mongo.get("/health")
    assert response.status_code == 503
    assert "Worker task failed" in response.json()["detail"]


def test_health_worker_stopped(mocker, client_with_mongo):
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = True
    mock_task.result.return_value = None
    mocker.patch.object(app.state, "worker_task", mock_task)

    response = client_with_mongo.get("/health")
    assert response.status_code == 503
    assert "Worker task stopped" in response.json()["detail"]


def test_health_mongo_unavailable(mocker, client_with_mongo, healthy_mongo_client):
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = False
    mocker.patch.object(app.state, "worker_task", mock_task)
    healthy_mongo_client.admin.command.side_effect = Exception("MongoDB unavailable")

    response = client_with_mongo.get("/health")
    assert response.status_code == 503
    assert "MongoDB unavailable" in response.json()["detail"]


def test_root(client_with_mongo):
    response = client_with_mongo.get("/")
    assert response.status_code == 404
