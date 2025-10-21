import contextvars
import logging

import fastapi
import starlette.middleware.base

from app import config

logger = logging.getLogger(__name__)

app_config = config.get_config()

ctx_trace_id = contextvars.ContextVar("trace_id")
ctx_request = contextvars.ContextVar("request")
ctx_response = contextvars.ContextVar("response")


# Inbound HTTP requests on the platform will have a `x-cdp-request-id` header.
# This can be used to follow a single request across multiple services.
# TraceIdMiddleware handles extracting the tracing header and persisting it
# for the duration of the request in the ContextVar `ctx_trace_id`.
class TraceIdMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    async def dispatch(self, request: fastapi.Request, call_next):
        req_trace_id = request.headers.get(app_config.tracing_header, None)
        if req_trace_id:
            ctx_trace_id.set(req_trace_id)

        ctx_request.set({"url": str(request.url), "method": request.method})

        response = await call_next(request)
        ctx_response.set({"status_code": response.status_code})
        return response
