import fastapi

router = fastapi.APIRouter(tags=["health"])


# Do not remove - used for health checks
@router.get("/health")
async def health():
    return {"status": "ok"}
