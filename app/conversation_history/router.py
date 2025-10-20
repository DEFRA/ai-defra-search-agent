from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.conversation_history.api_schemas import (
    ConversationHistoryResponse,
    MessageResponse,
    TokenUsageResponse,
)
from app.conversation_history.dependencies import get_conversation_history_service
from app.conversation_history.models import (
    ConversationNotFoundError,
)
from app.conversation_history.service import ConversationHistoryService

router = APIRouter(tags=["conversation-history"])


@router.get("/conversation-history/{conversation_id}")
async def get_conversation_history(
    conversation_id: UUID,
    history_service: ConversationHistoryService = Depends(get_conversation_history_service)
):
    try:
        conversation = await history_service.get_history(conversation_id)

        return ConversationHistoryResponse(
            conversation_id=str(conversation.conversation_id),
            messages=[
                MessageResponse(
                    role=message.role,
                    content=message.content
                )
                for message in conversation.messages
            ],
            token_usage=[
                TokenUsageResponse(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    model=usage.model,
                    stage_name=usage.stage_name
                )
                for usage in conversation.token_usage
            ]
        )
    except ConversationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
