from contextlib import asynccontextmanager
from logging import getLogger

import uvicorn
from fastapi import FastAPI

from app.chat.router import router as chat_router
from app.chat_with_history.router import router as chat_history_router
from app.common.mongo import get_mongo_client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.data_ingestion.router import router as data_router
from app.debug.router import router as debug_router
from app.health.router import router as health_router

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    client = await get_mongo_client()
    logger.info("MongoDB client connected")
    yield
    # Shutdown
    if client:
        await client.close()
        logger.info("MongoDB client closed")


app = FastAPI(lifespan=lifespan)

# Setup middleware
app.add_middleware(TraceIdMiddleware)

# Setup Routes
app.include_router(health_router)
app.include_router(data_router)
app.include_router(chat_history_router)
app.include_router(chat_router)
app.include_router(debug_router)


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_config=config.log_config,
        reload=config.python_env == "development",
    )


if __name__ == "__main__":
    main()
