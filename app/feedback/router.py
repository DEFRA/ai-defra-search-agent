import fastapi

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
    feedback_service: service.FeedbackService = fastapi.Depends(
        dependencies.get_feedback_service
    ),
):
    feedback = await feedback_service.submit_feedback(
        was_helpful=request.was_helpful,
        conversation_id=request.conversation_id,
        comment=request.comment,
    )

    return api_schemas.FeedbackResponse(
        feedback_id=feedback.id,
        timestamp=feedback.timestamp,
    )
