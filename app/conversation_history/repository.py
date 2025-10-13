from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import UUID

from pymongo.asynchronous.database import AsyncCollection, AsyncDatabase

from app.conversation_history.models import ChatMessage, ConversationHistory


class AbstractConversationHistoryRepository(ABC):
    @abstractmethod
    async def add_message(self, conversation_id: UUID, message: ChatMessage) -> None:
        pass

    @abstractmethod
    async def get_history(self, conversation_id: UUID) -> ConversationHistory | None:
        pass

    @abstractmethod
    async def create_conversation(self, conversation_id: UUID) -> ConversationHistory:
        pass


class MongoConversationHistoryRepository(AbstractConversationHistoryRepository):
    def __init__(self, db: AsyncDatabase):
        self.db: AsyncDatabase = db
        self.conversation_history: AsyncCollection = self.db.conversation_history

    async def add_message(self, conversation_id: UUID, message: ChatMessage) -> None:
        await self.conversation_history.update_one(
            {"conversationId": conversation_id},
            {
                "$push": {"messages": message.__dict__},
                "$setOnInsert": {"createdAt": datetime.now(UTC)},
            },
            upsert=True,
        )

    async def get_history(self, conversation_id: UUID) -> ConversationHistory | None:
        doc = await self.conversation_history.find_one({"conversationId": conversation_id})

        if doc:
            messages = [ChatMessage(**message) for message in doc.get("messages", [])]
            return ConversationHistory(conversation_id=conversation_id, messages=messages)

        return None

    async def create_conversation(self, conversation_id: UUID) -> ConversationHistory:
        await self.conversation_history.insert_one(
            {"conversationId": conversation_id, "messages": [], "createdAt": datetime.now(UTC)}
        )
        return ConversationHistory(conversation_id=conversation_id, messages=[])
