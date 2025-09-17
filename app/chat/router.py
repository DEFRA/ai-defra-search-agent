from logging import getLogger

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.lib.langgraph_rag_chat import run_rag_llm

logger = getLogger(__name__)

router = APIRouter(prefix="/langgraph/chat")
chat_history = []


class QuestionRequest(BaseModel):
    question: str = Field(
        description="The question to ask the model",
        examples=["What is machine learning?"],
    )


@router.post("/")
async def chat(request: QuestionRequest):
    try:
        question = request.question
        response = run_rag_llm(question)

        print("####### RESPONSE: ", response)

        # Check if response contains validation errors
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

    except ValidationError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.exception("Failed to chat with LangGraph Rag Chat")
        raise HTTPException(status_code=500, detail=str(e)) from e
