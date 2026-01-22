import asyncio
import json
import logging
import uuid

import fastapi
from fastapi import status
from sse_starlette.sse import EventSourceResponse

from app.chat import api_schemas, dependencies, models
from app.common.event_broker import get_event_broker
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    status_code=status.HTTP_200_OK,
    summary="Send a message to the chatbot with SSE streaming",
    description="Sends a user question to the specified model and streams the response status updates via Server-Sent Events.",
    responses={
        200: {"description": "SSE stream of message status updates"},
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

    # Queue message for processing
    async with sqs_client:
        await sqs_client.send_message(
            {
                "message_id": str(user_message.message_id),
                "conversation_id": str(conversation.id),
                "question": request.question,
                "model_id": request.model_id,
            }
        )

    # Stream message status updates via SSE
    async def event_generator():
        event_broker = get_event_broker()
        queue = await event_broker.subscribe(str(user_message.message_id))

        try:
            # Send initial queued status
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "message_id": str(user_message.message_id),
                        "conversation_id": str(conversation.id),
                        "status": models.MessageStatus.QUEUED.value,
                    }
                ),
            }

            # Stream updates from event broker
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield {
                        "event": "status",
                        "data": json.dumps(event),
                    }

                    # End stream after terminal status
                    if event.get("status") in [
                        models.MessageStatus.COMPLETED.value,
                        models.MessageStatus.FAILED.value,
                    ]:
                        break

                except TimeoutError:
                    # Send keepalive comment
                    yield {"event": "keepalive", "data": ""}

        except asyncio.CancelledError:
            logger.info(
                "SSE connection cancelled for message %s", user_message.message_id
            )
        finally:
            await event_broker.unsubscribe(str(user_message.message_id), queue)

    return EventSourceResponse(event_generator(), ping=None)


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation by ID",
    description="Retrieve a conversation with all its messages.",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    conversation_repository=fastapi.Depends(dependencies.get_conversation_repository),
):
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
