import uuid

import fastapi

from app.conversation_history import api_schemas, dependencies, models, service

router = fastapi.APIRouter(tags=["conversation-history"])


@router.get("/conversation-history/{conversation_id}")
async def get_conversation_history(
    conversation_id: uuid.UUID,
    history_service: service.ConversationHistoryService = fastapi.Depends(dependencies.get_conversation_history_service)
):
    try:
        conversation = await history_service.get_history(conversation_id)

        return api_schemas.ConversationHistoryResponse(
            conversation_id=str(conversation.conversation_id),
            messages=[
                api_schemas.MessageResponse(
                    role=message.role,
                    content=message.content,
                    sources=[
                        {
                            "content": doc.get("content", ""),
                            "name": doc.get("name", ""),
                            "location": doc.get("location", "")
                        }
                        for doc in message.context
                    ] if message.context else None
                )
                for message in conversation.messages
            ],
            token_usage=[
                api_schemas.TokenUsageResponse(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    model=usage.model,
                    stage_name=usage.stage_name
                )
                for usage in conversation.token_usage
            ]
        )
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e)) from None
