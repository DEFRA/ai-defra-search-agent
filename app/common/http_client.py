import logging

import httpx

from app import config
from app.common import tracing

logger = logging.getLogger(__name__)

app_config = config.get_config()

async def async_hook_request_tracing(request):
    trace_id = tracing.ctx_trace_id.get(None)
    if trace_id:
        request.headers[app_config.tracing_header] = trace_id


def hook_request_tracing(request):
    trace_id = tracing.ctx_trace_id.get(None)
    if trace_id:
        request.headers[app_config.tracing_header] = trace_id


def create_async_client(request_timeout: int = 30) -> httpx.AsyncClient:
    """
    Create an async HTTP client with configurable timeout.

    Args:
        request_timeout: Request timeout in seconds

    Returns:
        Configured httpx.AsyncClient instance
    """
    client_kwargs = {
        "timeout": request_timeout,
        "event_hooks": {"request": [async_hook_request_tracing]}
    }

    if app_config.http_proxy:
        logger.info("Using HTTP proxy: %s", app_config.http_proxy)

        proxy_mounts = {
            "http://": httpx.AsyncHTTPTransport(proxy=app_config.http_proxy),
            "https://": httpx.AsyncHTTPTransport(proxy=app_config.http_proxy)
        }

        client_kwargs["mounts"] = proxy_mounts

    return httpx.AsyncClient(**client_kwargs)


def create_client(request_timeout: int = 30) -> httpx.Client:
    """
    Create a sync HTTP client with configurable timeout.

    Args:
        request_timeout: Request timeout in seconds

    Returns:
        Configured httpx.Client instance
    """
    client_kwargs = {
        "timeout": request_timeout,
        "event_hooks": {"request": [hook_request_tracing]}
    }

    if app_config.http_proxy:
        logger.info("Using HTTP proxy: %s", app_config.http_proxy)

        proxy_mounts = {
            "http://": httpx.HTTPTransport(proxy=app_config.http_proxy),
            "https://": httpx.HTTPTransport(proxy=app_config.http_proxy)
        }

        client_kwargs["mounts"] = proxy_mounts

    return httpx.Client(**client_kwargs)
