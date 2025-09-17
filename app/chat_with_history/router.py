from datetime import UTC, datetime
from logging import getLogger
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError
from pymongo.collection import Collection

from app.common.mongo import get_db
from app.lib.llm_chat import run_llm

logger = getLogger(__name__)

router = APIRouter(prefix="/chat")
chat_history = []


class QuestionRequest(BaseModel):
    session_id: UUID = Field(
        description="The session ID for the chat (must be a valid GUID)",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    question: str = Field(
        description="The question to ask the model",
        examples=["What is machine learning?"],
    )


async def store_usage_data(
    db, question: str, answer: str, session_id: str, usage_data: dict
):
    usage_collection: Collection = db["user_usage"]

    for model_id, usage_info in usage_data.items():
        input_token_details = usage_info.get("input_token_details", {})

        usage_entry = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "model_id": model_id,
            "output_tokens": usage_info.get("output_tokens", 0),
            "input_tokens": usage_info.get("input_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "cache_creation": input_token_details.get("cache_creation", 0),
            "cache_read": input_token_details.get("cache_read", 0),
            "timestamp": datetime.now(tz=UTC),
        }

        await usage_collection.insert_one(usage_entry)
        logger.info("Stored usage data for model %s: %s", model_id, usage_entry)

        usage_entry.pop("_id", None)
        return usage_entry

    return None


@router.post("/")
async def chat(request: QuestionRequest, db=Depends(get_db)):
    try:
        chat_collection: Collection = db["chat_history"]

        session_id = str(request.session_id)

        chat_entry = await chat_collection.find_one({"session_id": session_id})
        chat_history = chat_entry["chat_history"] if chat_entry else []

        question = request.question
        response = run_llm(question, chat_history)

        print("LLM response: %s", response)

        answer = response["answer"]
        chat_history.append(("human", question))
        chat_history.append(("ai", answer))

        if "usage" in response:
            await store_usage_data(db, question, answer, session_id, response["usage"])

        if chat_entry:
            await chat_collection.update_one(
                {"session_id": session_id}, {"$set": {"chat_history": chat_history}}
            )
        else:
            await chat_collection.insert_one(
                {
                    "session_id": session_id,
                    "chat_history": chat_history,
                }
            )

        return {"status": "success", **response, "usage": response["usage"]}

    except ValidationError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.exception("Failed to chat with LangGraph Rag Chat")
        raise HTTPException(status_code=500, detail=str(e)) from e
