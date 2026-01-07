import fastapi

router = fastapi.APIRouter(tags=["health"])


# Do not remove - used for health checks
@router.get(
    "/health",
    summary="Health Check",
    description="Returns the operational status of the service.",
)
async def health():
    return {"status": "ok"}
