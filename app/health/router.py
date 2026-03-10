import asyncio
import logging
from typing import Annotated

import fastapi
import pymongo
from fastapi import status

from app.common import mongo

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["health"])


def get_worker_task(request: fastapi.Request) -> asyncio.Task | None:
    """Dependency to get worker task from app state."""
    return getattr(request.app.state, "worker_task", None)


# Do not remove - used for health checks
@router.get(
    "/health",
    summary="Health Check",
    description="Returns the operational status of the service.",
)
async def health(
    worker_task: Annotated[asyncio.Task | None, fastapi.Depends(get_worker_task)],
    mongo_client: Annotated[
        pymongo.AsyncMongoClient, fastapi.Depends(mongo.get_mongo_client)
    ],
):
    if worker_task is None:
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker task not running",
        )

    if worker_task.done():
        try:
            worker_task.result()
        except Exception as e:
            logger.error("Worker task failed: %s", e)
            raise fastapi.HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Worker task failed: {str(e)}",
            ) from None
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker task stopped",
        )

    try:
        await mongo_client.admin.command("ping")
    except Exception as e:
        logger.warning("MongoDB ping failed during health check")
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        ) from None

    return {"status": "ok"}
