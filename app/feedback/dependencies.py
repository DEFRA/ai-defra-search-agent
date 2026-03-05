import fastapi
import pymongo.asynchronous.database

from app import config, dependencies
from app.common import mongo
from app.feedback import repository, service


def get_feedback_repository(
    db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> repository.AbstractFeedbackRepository:
    return repository.MongoFeedbackRepository(
        db=db,
        retry_attempts=app_config.mongo.retry_attempts,
        retry_base_delay_seconds=app_config.mongo.retry_base_delay_seconds,
    )


def get_feedback_service(
    feedback_repository: repository.AbstractFeedbackRepository = fastapi.Depends(
        get_feedback_repository
    ),
) -> service.FeedbackService:
    return service.FeedbackService(feedback_repository=feedback_repository)
