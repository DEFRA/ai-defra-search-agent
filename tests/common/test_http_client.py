import httpx
import pytest

from app.common import http_client, tracing


def mock_handler(request):
    request_id = request.headers.get("x-cdp-request-id", "")
    return httpx.Response(200, text=request_id)


def test_async_client_factory_with_proxy_creates_client(mocker):
    mock_config = mocker.Mock()
    mock_config.http_proxy = "http://localhost:8888"
    mock_config.tracing_header = "x-cdp-request-id"

    client = http_client.create_async_client(mock_config, request_timeout=25)

    assert isinstance(client, httpx.AsyncClient)


def test_async_client_factory_no_proxy_creates_client(mocker):
    mock_config = mocker.Mock()
    mock_config.http_proxy = None
    mock_config.tracing_header = "x-cdp-request-id"

    client = http_client.create_async_client(mock_config, request_timeout=10)

    assert isinstance(client, httpx.AsyncClient)


def test_client_factory_with_proxy_creates_client(mocker):
    mock_config = mocker.Mock()
    mock_config.http_proxy = "http://localhost:8888"
    mock_config.tracing_header = "x-cdp-request-id"

    client = http_client.create_client(mock_config, request_timeout=20)

    assert isinstance(client, httpx.Client)


def test_client_factory_no_proxy_creates_client(mocker):
    mock_config = mocker.Mock()
    mock_config.http_proxy = None
    mock_config.tracing_header = "x-cdp-request-id"

    client = http_client.create_client(mock_config, request_timeout=15)

    assert isinstance(client, httpx.Client)


def test_trace_id_missing():
    tracing.ctx_trace_id.set("")
    hook = http_client.create_tracing_hook("x-cdp-request-id")
    client = httpx.Client(
        event_hooks={"request": [hook]},
        transport=httpx.MockTransport(mock_handler),
    )
    resp = client.get("http://localhost:1234/test")
    assert resp.text == ""


def test_trace_id_set():
    tracing.ctx_trace_id.set("trace-id-value")
    hook = http_client.create_tracing_hook("x-cdp-request-id")
    client = httpx.Client(
        event_hooks={"request": [hook]},
        transport=httpx.MockTransport(mock_handler),
    )
    resp = client.get("http://localhost:1234/test")
    assert resp.text == "trace-id-value"


@pytest.mark.asyncio
async def test_async_trace_id_set():
    tracing.ctx_trace_id.set("trace-id-value")
    hook = http_client.create_async_tracing_hook("x-cdp-request-id")

    async with httpx.AsyncClient(
        event_hooks={"request": [hook]},
        transport=httpx.MockTransport(mock_handler),
    ) as client:
        resp = await client.get("http://localhost:1234/test")
        assert resp.text == "trace-id-value"


@pytest.mark.asyncio
async def test_async_trace_id_missing():
    tracing.ctx_trace_id.set("")
    hook = http_client.create_async_tracing_hook("x-cdp-request-id")

    async with httpx.AsyncClient(
        event_hooks={"request": [hook]},
        transport=httpx.MockTransport(mock_handler),
    ) as client:
        resp = await client.get("http://localhost:1234/test")
        assert resp.text == ""
