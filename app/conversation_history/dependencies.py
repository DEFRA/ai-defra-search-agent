from fastapi import Depends

from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.conversation_history.repository import MongoConversationHistoryRepository
from app.conversation_history.service import ConversationHistoryService


def get_conversation_history_service(db: AsyncDatabase = Depends(get_db)) -> ConversationHistoryService:
    repository = MongoConversationHistoryRepository(db)
    return ConversationHistoryService(repository)
