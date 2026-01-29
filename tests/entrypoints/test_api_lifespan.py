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


@pytest.mark.asyncio
async def test_lifespan_closes_client_when_worker_cancelled(monkeypatch):
    """Ensure the mongo client is closed even if the worker task is cancelled.

    The lifecycle manager cancels the background worker and awaits it; when
    awaiting a cancelled task a `CancelledError` is raised by the task. The
    `finally` block must still run and close the mongo client.
    """

    from unittest.mock import AsyncMock

    # Create a dummy client with an awaitable close method we can assert on
    dummy_client = AsyncMock()

    async def fake_get_mongo_client(_cfg):
        return dummy_client

    monkeypatch.setattr(api, "mongo", api.mongo)
    monkeypatch.setattr(api.mongo, "get_mongo_client", fake_get_mongo_client)

    # Make the worker coroutine wait forever so cancelling it triggers CancelledError
    async def long_running_worker():
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "run_worker", long_running_worker)

    client = TestClient(api.app)

    # entering/exiting may raise a CancelledError when the worker task is
    # cancelled during shutdown. Accept that but ensure the mongo client's
    # close method was still awaited in the lifespan finally block.
    from concurrent.futures import CancelledError

    try:
        with client:
            assert hasattr(api.app.state, "worker_task")
    except CancelledError:
        # Some event-loop/thread combinations surface a CancelledError
        # from the background task cancellation; that's acceptable for
        # this test as long as the client.close() was awaited.
        pass

    # The dummy client's close() should have been awaited once during shutdown
    dummy_client.close.assert_awaited_once()
