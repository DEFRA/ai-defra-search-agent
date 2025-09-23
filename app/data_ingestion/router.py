from logging import getLogger

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from pydantic import BaseModel

from app.lib.store.vectorstore_client import VectorStoreClient

logger = getLogger(__name__)

router = APIRouter()


class SetupDataRequest(BaseModel):
    urls: list[str]


@router.get("/data/search")
async def search_data(
    query: str = Query(..., description="The query string for similarity search"),
    k: int = Query(1, description="Number of results to return"),
):
    """
    Perform a similarity search using the provided query string.
    """
    try:
        client = VectorStoreClient()
        results = client.similarity_search(query=query, k=k)
        logger.info("Similarity search results for query '%s': %s", query, results)

        return {"status": "success", "query": query, "results": results}

    except Exception as e:
        logger.exception("Failed to perform similarity search")
        raise HTTPException(status_code=500, detail=str(e)) from e


def setup_data_background(client: VectorStoreClient, urls: list[str]):
    doc_ids =client.load_documents(urls)
    logger.info("Loaded documents with ids: %s", doc_ids)


@router.post("/data/setup")
async def setup_data(request: SetupDataRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            setup_data_background,
            VectorStoreClient(),
            request.urls
        )

        return Response(
            status_code=202
        )

    except Exception as e:
        logger.exception("Failed to setup data")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/data/clear")
async def clear_data():
    try:
        client = VectorStoreClient()
        client.delete_documents()

        return {
            "status": "success",
        }

    except Exception as e:
        logger.exception("Failed to clear data")
        raise HTTPException(status_code=500, detail=str(e)) from e
