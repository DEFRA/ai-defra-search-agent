import logging

import fastapi

from app.chat import api_schemas, dependencies, models, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post("/chat", response_model=api_schemas.ChatResponse)
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
    except (models.UnsupportedModelError, ValueError) as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from None

    return api_schemas.ChatResponse(
        conversation_id=conversation.id,
        messages=[
            api_schemas.MessageResponse(
                role=message.role,
                content=message.content,
                modelName=message.model_name,
                model_id=message.model_id,
            )
            for message in conversation.messages
        ],
    )
