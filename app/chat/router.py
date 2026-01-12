import logging

import fastapi
from botocore.exceptions import ClientError
from fastapi import status

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
        400: {"description": "Bad request - unsupported model ID, invalid request data, or AWS Bedrock validation error"},
        404: {"description": "Conversation not found"},
        429: {"description": "Too many requests - rate limit exceeded"},
        500: {"description": "Internal server error from AWS Bedrock"},
        502: {"description": "Bad gateway - AWS service error"},
        503: {"description": "Service unavailable"},
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
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from None
    except UnsupportedModelError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None
    except ClientError as e:
        raise fastapi.HTTPException(
            status_code=e.response["ResponseMetadata"]["HTTPStatusCode"],
            detail=str(e),
        ) from None

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
