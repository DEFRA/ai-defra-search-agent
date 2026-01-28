import asyncio
from datetime import UTC, datetime, timedelta

import fastapi
from fastapi import status

router = fastapi.APIRouter(tags=["health"])

# Health check configuration
WORKER_HEARTBEAT_TIMEOUT_SECONDS = 90


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
    worker_task: asyncio.Task | None = fastapi.Depends(get_worker_task),
):
    from app.chat.worker import get_last_heartbeat

    # Check if worker task is running
    if worker_task is None or worker_task.done():
        if worker_task and worker_task.done():
            # Worker crashed - get the exception if any
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

    # Check worker heartbeat
    last_heartbeat = get_last_heartbeat()
    if last_heartbeat is None:
        # Worker just started, hasn't polled yet
        return {"status": "ok", "worker": "starting"}

    time_since_heartbeat = datetime.now(UTC) - last_heartbeat
    if time_since_heartbeat > timedelta(seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS):
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Worker heartbeat timeout: last seen {time_since_heartbeat.total_seconds():.0f}s ago",
        )

    return {"status": "ok"}
