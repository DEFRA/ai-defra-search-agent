from fastapi import Depends
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.lib.rag.rag_core import run_rag_llm_with_observability


async def get_rag_service_with_observability(db: AsyncDatabase = Depends(get_db)):
    """
    Dependency injection for RAG service with observability.
    Returns a callable that runs the RAG LLM with observability.
    """

    async def rag_service(query: str):
        return await run_rag_llm_with_observability(query, db)

    return rag_service
