import asyncio
from typing import Annotated

import fastapi
from fastapi import status

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
):
    if worker_task is None or worker_task.done():
        if worker_task and worker_task.done():
            try:
                worker_task.result()
            except Exception as e:
                raise fastapi.HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Worker task failed: {str(e)}",
                ) from None
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker task not running",
        )

    return {"status": "ok"}
