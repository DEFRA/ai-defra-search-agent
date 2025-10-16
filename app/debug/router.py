from logging import getLogger

from app.common.http_client import create_async_client, create_client
from fastapi import APIRouter

logger = getLogger(__name__)

router = APIRouter(tags=["chat"])

@router.get("/debug/async-proxy-test")
async def async_proxy_test():
    logger.info("Async proxy test endpoint called")

    async with create_async_client() as client:
        response = await client.get("https://www.gov.uk")
        response.raise_for_status()

        return { "status": "success", "content": response.text }


@router.get("/debug/sync-proxy-test")
def sync_proxy_test():
    logger.info("Sync proxy test endpoint called")

    with create_client() as client:
        response = client.get("https://www.gov.uk")
        response.raise_for_status()

        return { "status": "success", "content": response.text }
