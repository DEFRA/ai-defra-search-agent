import httpx

from app.common import http_client, tracing


def mock_handler(request):
    request_id = request.headers.get("x-cdp-request-id", "")
    return httpx.Response(200, text=request_id)


def test_async_client_factory_with_proxy_creates_client():
    original_proxy = http_client.app_config.http_proxy

    http_client.app_config.http_proxy = "http://localhost:8888"

    client = http_client.create_async_client(request_timeout=25)

    assert isinstance(client, httpx.AsyncClient)

    http_client.app_config.http_proxy = original_proxy


def test_async_client_factory_no_proxy_creates_client():
    original_proxy = http_client.app_config.http_proxy

    http_client.app_config.http_proxy = None

    client = http_client.create_async_client(request_timeout=10)

    assert isinstance(client, httpx.AsyncClient)

    http_client.app_config.http_proxy = original_proxy


def test_client_factory_with_proxy_creates_client():
    original_proxy = http_client.app_config.http_proxy

    http_client.app_config.http_proxy = "http://localhost:8888"

    client = http_client.create_client(request_timeout=20)

    assert isinstance(client, httpx.Client)

    http_client.app_config.http_proxy = original_proxy


def test_client_factory_no_proxy_creates_client():
    original_proxy = http_client.app_config.http_proxy

    http_client.app_config.http_proxy = None

    client = http_client.create_client(request_timeout=15)

    assert isinstance(client, httpx.Client)

    http_client.app_config.http_proxy = original_proxy


def test_trace_id_missing():
    tracing.ctx_trace_id.set("")
    client = httpx.Client(
        event_hooks={"request": [http_client.hook_request_tracing]},
        transport=httpx.MockTransport(mock_handler),
    )
    resp = client.get("http://localhost:1234/test")
    assert resp.text == ""


def test_trace_id_set():
    tracing.ctx_trace_id.set("trace-id-value")
    client = httpx.Client(
        event_hooks={"request": [http_client.hook_request_tracing]},
        transport=httpx.MockTransport(mock_handler),
    )
    resp = client.get("http://localhost:1234/test")
    assert resp.text == "trace-id-value"
