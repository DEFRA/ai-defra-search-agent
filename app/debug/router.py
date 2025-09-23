from logging import getLogger

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.lib.store.vectorstore_client import VectorStoreClient

logger = getLogger(__name__)

router = APIRouter(prefix="/debug")
chat_history = []


class QuestionRequest(BaseModel):
    question: str = Field(
        description="The question to ask the model",
        examples=["What is machine learning?"],
    )


@router.post("/vectorstore/chat")
async def chat(request: QuestionRequest):
    try:
        question = request.question
        client = VectorStoreClient()
        retriever = client.as_retriever()
        retrieved_docs = retriever.invoke(question)

        print("####### RESPONSE: ", retrieved_docs)

        return {"status": "success", "retrieved_docs": retrieved_docs}

    except ValidationError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.exception("Failed to chat with Debug Vector Store Chat")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/vectorstore/count")
async def get_document_count():
    try:
        client = VectorStoreClient()
        count = client.get_document_count()
        has_content = client.has_content()

        return {
            "status": "success",
            "document_count": count,
            "has_content": has_content,
            "message": f"Vector store contains {count} document chunks"
            if count > 0
            else "Vector store is empty",
        }
    except Exception as e:
        logger.exception("Failed to get document count")
        raise HTTPException(status_code=500, detail=str(e)) from e

