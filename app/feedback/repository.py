import abc
import uuid

import pymongo.asynchronous.database

from app.common import mongo
from app.feedback import models


class AbstractFeedbackRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, feedback: models.Feedback) -> None:
        """Save the feedback to the repository."""

    @abc.abstractmethod
    async def get(self, feedback_id: uuid.UUID) -> models.Feedback | None:
        """Get the feedback from the repository."""


class MongoFeedbackRepository(AbstractFeedbackRepository):
    def __init__(
        self,
        db: pymongo.asynchronous.database.AsyncDatabase,
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.5,
    ):
        self.db = db
        self.retry_attempts = retry_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.feedback: pymongo.asynchronous.collection.AsyncCollection = (
            self.db.feedback
        )

    def _retry(self, operation):
        return mongo.retry_mongo_operation(
            operation, self.retry_attempts, self.retry_base_delay_seconds
        )

    async def save(self, feedback: models.Feedback) -> None:
        async def _op():
            await self.feedback.update_one(
                {"feedback_id": feedback.id},
                {
                    "$set": {
                        "feedback_id": feedback.id,
                        "conversation_id": feedback.conversation_id,
                        "was_helpful": feedback.was_helpful,
                        "comment": feedback.comment,
                        "timestamp": feedback.timestamp,
                    }
                },
                upsert=True,
            )

        await self._retry(_op)

    async def get(self, feedback_id: uuid.UUID) -> models.Feedback | None:
        async def _op():
            feedback_doc = await self.feedback.find_one({"feedback_id": feedback_id})

            if not feedback_doc:
                return None

            return models.Feedback(
                id=feedback_doc["feedback_id"],
                conversation_id=feedback_doc.get("conversation_id"),
                was_helpful=models.WasHelpfulRating(feedback_doc["was_helpful"]),
                comment=feedback_doc.get("comment"),
                timestamp=feedback_doc["timestamp"],
            )

        return await self._retry(_op)
