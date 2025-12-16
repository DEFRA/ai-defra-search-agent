import fastapi
import pymongo.asynchronous.database

from app.common import mongo
from app.feedback import repository, service


def get_feedback_repository(
    db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db),
) -> repository.AbstractFeedbackRepository:
    return repository.MongoFeedbackRepository(db=db)


def get_feedback_service(
    feedback_repository: repository.AbstractFeedbackRepository = fastapi.Depends(
        get_feedback_repository
    ),
) -> service.FeedbackService:
    return service.FeedbackService(feedback_repository=feedback_repository)
