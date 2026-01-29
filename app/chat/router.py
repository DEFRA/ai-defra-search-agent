import json
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
    description="Sends a user question to the specified model and queues it for processing. Returns conversation and message IDs for streaming via GET /chat/stream/{message_id}.",
    responses={
        202: {"description": "Message queued successfully"},
        400: {
            "description": "Bad request - unsupported model ID, invalid request data, or AWS Bedrock validation error"
        },
    },
)
async def chat(
    request: api_schemas.ChatRequest,
    conversation_repository=fastapi.Depends(dependencies.get_conversation_repository),
    sqs_client=fastapi.Depends(dependencies.get_sqs_client),
    model_resolution_service=fastapi.Depends(dependencies.get_model_resolution_service),
):
    """Accept a chat request and enqueue it for asynchronous processing.

    The endpoint validates the requested model synchronously, persists a
    conversation/message with status `QUEUED` and sends a job message to the
    configured SQS queue. The response contains the `conversation_id` and
    `message_id` so clients can stream or poll for updates.
    """

    # Validate model immediately
    try:
        resolved_model = model_resolution_service.resolve_model(request.model_id)
    except UnsupportedModelError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None

    # Create user message with QUEUED status
    user_message = models.UserMessage(
        content=request.question,
        model_id=request.model_id,
        model_name=resolved_model.name if resolved_model else request.model_id,
        status=models.MessageStatus.QUEUED,
    )

    # Get or create conversation
    if request.conversation_id:
        conversation = await conversation_repository.get(request.conversation_id)
        if not conversation:
            raise fastapi.HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )
        conversation.add_message(user_message)
    else:
        conversation = models.Conversation(messages=[user_message])

    # Save conversation with queued message
    await conversation_repository.save(conversation)

    # Queue message for processing (serialize at call site so SQSClient
    # remains a transport-only helper)
    async with sqs_client:
        await sqs_client.send_message(
            json.dumps(
                {
                    "message_id": str(user_message.message_id),
                    "conversation_id": str(conversation.id),
                    "question": request.question,
                    "model_id": request.model_id,
                }
            )
        )

    # Return immediately with IDs for client to open SSE stream
    return {
        "message_id": str(user_message.message_id),
        "conversation_id": str(conversation.id),
        "status": models.MessageStatus.QUEUED.value,
    }


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation by ID",
    description="Retrieve a conversation with all its messages.",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    conversation_repository=fastapi.Depends(dependencies.get_conversation_repository),
):
    """Retrieve a conversation by its UUID.

    Returns a JSON serialisable dictionary containing the conversation ID
    and a list of messages with their status and payload fields. Raises
    HTTP 404 if the conversation is not found.
    """

    conversation = await conversation_repository.get(conversation_id)
    if conversation is None:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
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
