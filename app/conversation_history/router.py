from fastapi import APIRouter, Depends, HTTPException

from app.conversation_history.models import ConversationHistory, ConversationNotFoundError
from app.conversation_history.dependencies import get_conversation_history_service
from app.conversation_history.service import ConversationHistoryService

router = APIRouter(tags=["conversation-history"])


@router.get("/conversation-history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    history_service: ConversationHistoryService = Depends(get_conversation_history_service)
):
    try:
        history = await history_service.get_history(conversation_id)

        return history
    except ConversationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
