import logging
from typing import Any

import httpx

from app import config
from app.common import tracing

logger = logging.getLogger(__name__)


def create_async_tracing_hook(tracing_header: str):
    async def async_hook_request_tracing(request):
        trace_id = tracing.ctx_trace_id.get(None)
        if trace_id:
            request.headers[tracing_header] = trace_id

    return async_hook_request_tracing


def create_tracing_hook(tracing_header: str):
    def hook_request_tracing(request):
        trace_id = tracing.ctx_trace_id.get(None)
        if trace_id:
            request.headers[tracing_header] = trace_id

    return hook_request_tracing


def create_async_client(
    app_config: config.AppConfig, request_timeout: int = 30
) -> httpx.AsyncClient:
    """
    Create an async HTTP client with configurable timeout.

    Args:
        app_config: Application configuration
        request_timeout: Request timeout in seconds

    Returns:
        Configured httpx.AsyncClient instance
    """
    client_kwargs: dict[str, Any] = {
        "timeout": request_timeout,
        "event_hooks": {
            "request": [create_async_tracing_hook(app_config.tracing_header)]
        },
    }

    if app_config.http_proxy:
        logger.info("Using HTTP proxy: %s", app_config.http_proxy)

        proxy_mounts = {
            "http://": httpx.AsyncHTTPTransport(proxy=app_config.http_proxy),
            "https://": httpx.AsyncHTTPTransport(proxy=app_config.http_proxy),
        }

        client_kwargs["mounts"] = proxy_mounts

    return httpx.AsyncClient(**client_kwargs)


def create_client(
    app_config: config.AppConfig, request_timeout: int = 30
) -> httpx.Client:
    """
    Create a sync HTTP client with configurable timeout.

    Args:
        app_config: Application configuration
        request_timeout: Request timeout in seconds

    Returns:
        Configured httpx.Client instance
    """
    client_kwargs: dict[str, Any] = {
        "timeout": request_timeout,
        "event_hooks": {"request": [create_tracing_hook(app_config.tracing_header)]},
    }

    if app_config.http_proxy:
        logger.info("Using HTTP proxy: %s", app_config.http_proxy)

        proxy_mounts = {
            "http://": httpx.HTTPTransport(proxy=app_config.http_proxy),
            "https://": httpx.HTTPTransport(proxy=app_config.http_proxy),
        }

        client_kwargs["mounts"] = proxy_mounts

    return httpx.Client(**client_kwargs)
