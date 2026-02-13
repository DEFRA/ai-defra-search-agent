import asyncio
import contextlib
import logging

import fastapi
import fastapi.exceptions
import uvicorn

from app import config
from app.chat import router as chat_router
from app.chat.worker import run_worker
from app.common import mongo, tracing
from app.feedback import router as feedback_router
from app.health import router as health_router
from app.models import UnsupportedModelError
from app.models import router as models_router

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    app_config = config.get_config()
    client = await mongo.get_mongo_client(app_config)
    logger.info("MongoDB client connected")

    app.state.worker_task = asyncio.create_task(run_worker())
    logger.info("Worker task started")

    yield

    worker_task = app.state.worker_task
    if worker_task and not worker_task.done():
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        logger.info("Worker task stopped")

    if client:
        await asyncio.shield(client.close())
        logger.info("MongoDB client closed")


app = fastapi.FastAPI(
    title="AI Defra Search Agent",
    description="API for the AI-powered search agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def validation_exception_handler(
    _: fastapi.Request, exc: fastapi.exceptions.RequestValidationError
):
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


@app.exception_handler(UnsupportedModelError)
async def unsupported_model_exception_handler(
    _: fastapi.Request, exc: UnsupportedModelError
):
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


app.add_middleware(tracing.TraceIdMiddleware)

app.include_router(health_router.router)
app.include_router(models_router.router)
app.include_router(chat_router.router)
app.include_router(feedback_router.router)


def main() -> None:  # pragma: no cover
    app_config = config.get_config()
    uvicorn.run(
        "app.entrypoints.api:app",
        host=app_config.host,
        port=app_config.port,
        log_config=app_config.log_config,
        reload=app_config.python_env == "development",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
