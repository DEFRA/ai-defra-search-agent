import logging

import fastapi

from app.chat import api_schemas, dependencies, models, service
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=api_schemas.ChatResponse,
    summary="Send a message to the chatbot",
    description="Sends a user question to the specified model and retrieves the response along with the conversation history.",
    responses={
        404: {"description": "Conversation not found"},
        400: {"description": "Unsupported model ID"},
    },
)
async def chat(
    request: api_schemas.ChatRequest,
    chat_service: service.ChatService = fastapi.Depends(dependencies.get_chat_service),
):
    try:
        conversation = await chat_service.execute_chat(
            question=request.question,
            model_id=request.model_id,
            conversation_id=request.conversation_id,
        )
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e)) from None
    except UnsupportedModelError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from None

    return api_schemas.ChatResponse(
        conversation_id=conversation.id,
        messages=[
            api_schemas.MessageResponse(
                role=message.role,
                content=message.content,
                model_name=message.model_name,
                model_id=message.model_id,
                timestamp=message.timestamp,
            )
            for message in conversation.messages
        ],
    )
