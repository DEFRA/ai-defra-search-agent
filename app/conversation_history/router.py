import logging
import uuid

import fastapi

from app.conversation_history import api_schemas, dependencies, models, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["conversation-history"])


@router.get("/conversation-history/{conversation_id}")
async def get_conversation_history(
    conversation_id: uuid.UUID,
    history_service: service.ConversationHistoryService = fastapi.Depends(
        dependencies.get_conversation_history_service
    ),
):
    try:
        conversation = await history_service.get_history(conversation_id)

        return api_schemas.ConversationHistoryResponse(
            conversation_id=str(conversation.conversation_id),
            messages=[
                api_schemas.MessageResponse(role=message.role, content=message.content)
                for message in conversation.messages
            ],
            token_usage=[
                api_schemas.TokenUsageResponse(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    model=usage.model,
                    stage_name=usage.stage_name,
                )
                for usage in conversation.token_usage
            ],
        )
    except models.ConversationNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e)) from None


@router.post("/conversation-history/token-usage")
async def get_bulk_token_usage(
    request: api_schemas.BulkTokenUsageRequest,
    history_service: service.ConversationHistoryService = fastapi.Depends(
        dependencies.get_conversation_history_service
    ),
) -> api_schemas.BulkTokenUsageResponse:
    try:
        conversation_ids = [uuid.UUID(conv_id) for conv_id in request.conversation_ids]
        usage_data = await history_service.get_bulk_token_usage(conversation_ids)

        logger.info("Bulk token usage data: %s", usage_data)

        return api_schemas.BulkTokenUsageResponse(
            overall_usage=api_schemas.TokenUsageSummary(
                total_input_tokens=usage_data["overall_usage"]["total_input_tokens"],
                total_output_tokens=usage_data["overall_usage"]["total_output_tokens"],
                total_tokens=usage_data["overall_usage"]["total_tokens"],
            ),
            usage_by_model=[
                api_schemas.ModelTokenUsageSummary(
                    model=model_usage["model"],
                    total_input_tokens=model_usage["total_input_tokens"],
                    total_output_tokens=model_usage["total_output_tokens"],
                    total_tokens=model_usage["total_tokens"],
                )
                for model_usage in usage_data["usage_by_model"]
            ],
            usage_by_conversation=[
                api_schemas.ConversationTokenUsageSummary(
                    conversation_id=conv_usage["conversation_id"],
                    total_input_tokens=conv_usage["total_input_tokens"],
                    total_output_tokens=conv_usage["total_output_tokens"],
                    total_tokens=conv_usage["total_tokens"],
                    models=[
                        api_schemas.ModelTokenUsageSummary(
                            model=model_usage["model"],
                            total_input_tokens=model_usage["total_input_tokens"],
                            total_output_tokens=model_usage["total_output_tokens"],
                            total_tokens=model_usage["total_tokens"],
                        )
                        for model_usage in conv_usage["models"]
                    ],
                )
                for conv_usage in usage_data["usage_by_conversation"]
            ],
        )
    except ValueError as e:
        raise fastapi.HTTPException(
            status_code=400, detail=f"Invalid conversation ID format: {str(e)}"
        ) from None
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from None
