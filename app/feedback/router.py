import logging

import fastapi

from app.feedback import api_schemas, dependencies, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=api_schemas.FeedbackResponse, status_code=201)
async def submit_feedback(
    request: api_schemas.FeedbackRequest,
    feedback_service: service.FeedbackService = fastapi.Depends(
        dependencies.get_feedback_service
    ),
):
    try:
        feedback = await feedback_service.submit_feedback(
            was_helpful=request.was_helpful,
            conversation_id=request.conversation_id,
            comment=request.comment,
        )

        logger.info(
            {
                "feedback_id": feedback.id,
                "conversation_id": feedback.conversation_id,
                "was_helpful": feedback.was_helpful,
            },
            "Feedback submitted successfully",
        )

        return api_schemas.FeedbackResponse(
            feedback_id=feedback.id,
            timestamp=feedback.timestamp,
        )
    except Exception as e:
        logger.error({"error": str(e)}, "Error submitting feedback")
        raise fastapi.HTTPException(
            status_code=500, detail="An error occurred while submitting feedback."
        ) from e
