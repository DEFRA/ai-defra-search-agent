from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.lib.rag.enhanced_langgraph_rag import run_rag_llm_with_observability
from app.lib.rag.langgraph_rag_chat import run_rag_llm

logger = getLogger(__name__)

router = APIRouter(tags=["chat"])


class QuestionRequest(BaseModel):
    question: str = Field(
        description="The question to ask the model",
        examples=[
            "What ethical consideration do we need to make sure we cover using AI?"
        ],
    )
    conversation_id: str | None = Field(
        default=None,
        description="The conversation ID for tracking history. Leave empty to start a new conversation.",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )


@router.post("/langgraph/chat")
async def chat(request: QuestionRequest):
    try:
        question = request.question
        response = run_rag_llm(question)

        if "validation_error" in response:
            return {
                "status": "validation_failed",
                "error": response["validation_error"],
                "severity": response.get("validation_severity", "medium"),
                **{
                    k: v
                    for k, v in response.items()
                    if k not in ["validation_error", "validation_severity"]
                },
            }

        return {"status": "success", **response}

    # except ValidationError as e:
    #    logger.error("Validation error: %s", e)
    #    raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.exception("Failed to chat with LangGraph Rag Chat")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/langgraph/chat/enhanced")
async def chat_with_observability(
    request: QuestionRequest, db: AsyncDatabase = Depends(get_db)
):
    try:
        question = request.question
        conversation_id = request.conversation_id
        response = await run_rag_llm_with_observability(question, db, conversation_id)

        logger.info(
            "Enhanced RAG chat completed",
            extra={
                "execution_id": response.get("execution_id"),
                "question_preview": question[:100],
                "has_documents": len(response.get("source_documents", [])) > 0,
                "token_usage": response.get("usage", {}),
            },
        )

        if "validation_error" in response:
            return {
                "status": "validation_failed",
                "error": response["validation_error"],
                "severity": response.get("validation_severity", "medium"),
                **{
                    k: v
                    for k, v in response.items()
                    if k not in ["validation_error", "validation_severity"]
                },
            }

        if "error" in response:
            return {
                "status": "system_error",
                "error": response["error"],
                "execution_id": response.get("execution_id"),
                **{k: v for k, v in response.items() if k not in ["error"]},
            }

        return {
            "status": "success",
            "observability": {
                "execution_id": response.get("execution_id"),
                "tracked": True,
                "monitoring": "enabled",
            },
            **response,
        }

    # except ValidationError as e:
    #    logger.error("Validation error in enhanced endpoint: %s", e)
    #    raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.exception("Failed to chat with enhanced LangGraph Rag Chat")
        raise HTTPException(status_code=500, detail=str(e)) from e
