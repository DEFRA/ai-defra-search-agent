from logging import getLogger

from fastapi import APIRouter, HTTPException, Query
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


@router.post("/data/setup")
async def setup_data(request: SetupDataRequest):
    try:
        client = VectorStoreClient()
        doc_ids = client.load_documents(request.urls)
        logger.info("Loaded documents with ids: %s", doc_ids)
        return {
            "status": "success",
            "docIds": doc_ids,
        }

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
