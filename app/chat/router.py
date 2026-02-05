import logging
import uuid

import fastapi
from fastapi import status

from app.chat import api_schemas, dependencies, models
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a message to the chatbot",
    description="Queues a user question for asynchronous processing. Poll GET /conversations/{conversation_id} to retrieve responses.",
    responses={
        202: {"description": "Message queued successfully"},
        400: {
            "description": "Bad request - unsupported model ID, invalid request data, or AWS Bedrock validation error"
        },
        404: {"description": "Conversation not found"},
    },
)
async def chat(
    request: api_schemas.ChatRequest,
    chat_service=fastapi.Depends(dependencies.get_chat_service),
):
    """Queue a chat request and return message/conversation IDs for streaming."""
    try:
        message_id, conversation_id, status_value = await chat_service.queue_chat(
            question=request.question,
            model_id=request.model_id,
            conversation_id=request.conversation_id,
        )
    except UnsupportedModelError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from None

    return {
        "message_id": str(message_id),
        "conversation_id": str(conversation_id),
        "status": status_value.value,
    }


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation by ID",
    description="Retrieve a conversation with all its messages.",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    chat_service=fastapi.Depends(dependencies.get_chat_service),
):
    """Retrieve a conversation with all its messages."""
    try:
        conversation = await chat_service.get_conversation(conversation_id)
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from None

    return {
        "conversation_id": str(conversation.id),
        "messages": [
            {
                "message_id": str(msg.message_id),
                "role": msg.role,
                "content": msg.content,
                "model_id": msg.model_id,
                "model_name": msg.model_name,
                "status": msg.status.value,
                "error_message": msg.error_message,
                "error_code": msg.error_code,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in conversation.messages
        ],
    }
