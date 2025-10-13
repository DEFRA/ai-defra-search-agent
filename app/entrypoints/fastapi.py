from contextlib import asynccontextmanager
from logging import getLogger

import uvicorn
from fastapi import FastAPI

from app.common.mongo import get_mongo_client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.health.router import router as health_router
from app.v2_chat.router import router as chat_v2_router

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
app.include_router(chat_v2_router, prefix="/v2")

# app.include_router(conversation_history_router)


def main() -> None:
    uvicorn.run(
        "app.entrypoints.fastapi:app",
        host=config.host,
        port=config.port,
        log_config=config.log_config,
        reload=config.python_env == "development",
    )


if __name__ == "__main__":
    main()
