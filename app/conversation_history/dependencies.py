import fastapi
import pymongo.asynchronous.database

from app.common import mongo
from app.conversation_history import repository, service


def get_conversation_history_service(db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db)) -> service.ConversationHistoryService:
    repo = repository.MongoConversationHistoryRepository(db)
    return service.ConversationHistoryService(repo)
