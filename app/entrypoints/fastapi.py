import contextlib
import logging

import fastapi
import fastapi.exceptions
import uvicorn

from app import config
from app.chat import router as chat_router
from app.common import mongo, tracing
from app.health import router as health_router

logger = logging.getLogger(__name__)

app_config = config.get_config()


@contextlib.asynccontextmanager
async def lifespan(_: fastapi.FastAPI):
    # Startup
    client = await mongo.get_mongo_client()
    logger.info("MongoDB client connected")
    yield
    # Shutdown
    if client:
        await client.close()
        logger.info("MongoDB client closed")


app = fastapi.FastAPI(lifespan=lifespan)

@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: fastapi.exceptions.RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

# Setup middleware
app.add_middleware(tracing.TraceIdMiddleware)

# Setup Routes
app.include_router(health_router.router)
app.include_router(chat_router.router)

def main() -> None:
    uvicorn.run(
        "app.entrypoints.fastapi:app",
        host=app_config.host,
        port=app_config.port,
        log_config=app_config.log_config,
        reload=app_config.python_env == "development",
    )


if __name__ == "__main__":
    main()
