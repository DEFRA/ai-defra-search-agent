import logging
import uuid

from app.feedback import models, repository

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, feedback_repository: repository.AbstractFeedbackRepository):
        self.feedback_repository = feedback_repository

    async def submit_feedback(
        self,
        was_helpful: bool,
        conversation_id: uuid.UUID | None = None,
        comment: str | None = None,
    ) -> models.Feedback:
        feedback = models.Feedback(
            conversation_id=conversation_id,
            was_helpful=was_helpful,
            comment=comment,
        )

        await self.feedback_repository.save(feedback)
        logger.info(
            "Feedback submitted successfully",
            extra={
                "feedback_id": str(feedback.id),
                "conversation_id": str(feedback.conversation_id),
            },
        )

        return feedback
