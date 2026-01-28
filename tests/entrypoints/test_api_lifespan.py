import asyncio

import pytest
from fastapi.testclient import TestClient

from app.entrypoints import api


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_worker(monkeypatch):
    # Patch mongo client creation to avoid real DB
    async def fake_get_mongo_client(_cfg):
        class Dummy:
            async def close(self):
                return None

        return Dummy()

    monkeypatch.setattr(api, "mongo", api.mongo)
    monkeypatch.setattr(api.mongo, "get_mongo_client", fake_get_mongo_client)

    # Patch run_worker to a coroutine that completes quickly
    async def fake_run_worker():
        await asyncio.sleep(0)

    monkeypatch.setattr(api, "run_worker", fake_run_worker)

    client = TestClient(api.app)

    # entering the context should start lifespan (startup)
    with client:
        # worker_task should exist on app state
        assert hasattr(api.app.state, "worker_task")
        task = api.app.state.worker_task
        assert task is not None

    # after exiting, task should be done or cancelled
    assert api.app.state.worker_task.done()
