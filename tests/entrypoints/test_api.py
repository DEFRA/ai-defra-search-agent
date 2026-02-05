from fastapi.testclient import TestClient

from app.entrypoints.api import app

client = TestClient(app)


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


def test_health(mocker):
    # Mock worker_task in app state
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = False
    mocker.patch.object(app.state, "worker_task", mock_task)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_worker_not_running(mocker):
    # Mock worker_task as None (not started)
    mocker.patch.object(app.state, "worker_task", None)

    response = client.get("/health")
    assert response.status_code == 503
    assert "Worker task not running" in response.json()["detail"]


def test_health_worker_crashed(mocker):
    # Mock worker_task as done (crashed)
    mock_task = mocker.MagicMock()
    mock_task.done.return_value = True
    mock_task.result.side_effect = Exception("Worker crashed")
    mocker.patch.object(app.state, "worker_task", mock_task)

    response = client.get("/health")
    assert response.status_code == 503
    assert "Worker task failed" in response.json()["detail"]


def test_root():
    response = client.get("/")
    assert response.status_code == 404
