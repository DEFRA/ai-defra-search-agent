from typing import Annotated

import fastapi
from fastapi import status

from app.common.mongo import MongoUnavailableError
from app.feedback import api_schemas, dependencies, service

router = fastapi.APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=api_schemas.FeedbackResponse,
    status_code=201,
    summary="Submit feedback",
    description="Allows users to submit feedback (helpful/not helpful) and optional comments for a specific conversation.",
)
async def submit_feedback(
    request: api_schemas.FeedbackRequest,
    feedback_service: Annotated[
        service.FeedbackService, fastapi.Depends(dependencies.get_feedback_service)
    ],
):
    try:
        feedback = await feedback_service.submit_feedback(
            was_helpful=request.was_helpful,
            conversation_id=request.conversation_id,
            comment=request.comment,
        )
    except MongoUnavailableError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        ) from None

    return api_schemas.FeedbackResponse(
        feedback_id=feedback.id,
        timestamp=feedback.timestamp,
    )
