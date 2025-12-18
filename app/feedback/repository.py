import abc
import uuid

import pymongo.asynchronous.database

from app.feedback import models


class AbstractFeedbackRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, feedback: models.Feedback) -> None:
        """Save the feedback to the repository."""

    @abc.abstractmethod
    async def get(self, feedback_id: uuid.UUID) -> models.Feedback | None:
        """Get the feedback from the repository."""


class MongoFeedbackRepository(AbstractFeedbackRepository):
    def __init__(self, db: pymongo.asynchronous.database.AsyncDatabase):
        self.db = db
        self.feedback: pymongo.asynchronous.collection.AsyncCollection = (
            self.db.feedback
        )

    async def save(self, feedback: models.Feedback) -> None:
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

    async def get(self, feedback_id: uuid.UUID) -> models.Feedback | None:
        feedback_doc = await self.feedback.find_one({"feedback_id": feedback_id})

        if not feedback_doc:
            return None

        return models.Feedback(
            id=feedback_doc["feedback_id"],
            conversation_id=feedback_doc.get("conversation_id"),
            was_helpful=feedback_doc["was_helpful"],
            comment=feedback_doc.get("comment"),
            timestamp=feedback_doc["timestamp"],
        )
