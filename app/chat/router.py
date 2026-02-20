import logging
import uuid
from typing import Annotated

import fastapi
from fastapi import status

from app.chat import api_schemas, dependencies, models, service
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a message to the chatbot",
    description="Queues a user question for asynchronous processing. Poll GET /conversations/{conversation_id} to retrieve responses.",
    response_model=api_schemas.QueueChatResponse,
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
    chat_service: Annotated[
        service.ChatService, fastapi.Depends(dependencies.get_queue_chat_service)
    ],
):
    """Queue a chat request and return message/conversation IDs."""
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

    return api_schemas.QueueChatResponse(
        message_id=message_id,
        conversation_id=conversation_id,
        status=status_value,
    )


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation by ID",
    description="Retrieve a conversation with all its messages.",
    response_model=api_schemas.ChatResponse,
)
async def get_conversation(
    conversation_id: uuid.UUID,
    chat_service: Annotated[
        service.ChatService, fastapi.Depends(dependencies.get_queue_chat_service)
    ],
):
    """Retrieve a conversation with all its messages."""
    try:
        conversation = await chat_service.get_conversation(conversation_id)
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from None

    messages = [
        api_schemas.MessageResponse(
            message_id=msg.message_id,
            role=msg.role,
            content=msg.content,
            model_id=msg.model_id,
            model_name=msg.model_name,
            status=msg.status.value
            if isinstance(msg, models.UserMessage)
            else models.MessageStatus.COMPLETED.value,
            error_message=msg.error_message
            if isinstance(msg, models.UserMessage)
            else None,
            timestamp=msg.timestamp,
        )
        for msg in conversation.messages
    ]

    return api_schemas.ChatResponse(
        conversation_id=conversation.id,
        messages=messages,
    )
