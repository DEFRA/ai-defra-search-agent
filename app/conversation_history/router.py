from fastapi import APIRouter, Depends, HTTPException
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.lib.conversation_history.service import ConversationHistoryService

router = APIRouter(prefix="/conversation-history")


@router.get("/{conversation_id}")
async def get_conversation_history(
    conversation_id: str, db: AsyncDatabase = Depends(get_db)
):
    service = ConversationHistoryService(db)
    history = await service.get_history(conversation_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if "_id" in history:
        history["_id"] = str(history["_id"])
    return history
