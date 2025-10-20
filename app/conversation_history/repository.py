from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import UUID

from app.v2_chat.models import StageTokenUsage
from pymongo.asynchronous.database import AsyncCollection, AsyncDatabase

from app.conversation_history.models import ChatMessage, ConversationHistory


class AbstractConversationHistoryRepository(ABC):
    @abstractmethod
    async def add_message(self, conversation_id: UUID, message: ChatMessage) -> None:
        pass

    @abstractmethod
    async def add_token_usage(self, conversation_id: UUID, token_usage: StageTokenUsage) -> None:
        pass

    @abstractmethod
    async def reset_token_usage(self, conversation_id: UUID) -> None:
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
        self.conversation_history: AsyncCollection = self.db.conversationHistory

    async def add_message(self, conversation_id: UUID, message: ChatMessage) -> None:
        await self.conversation_history.update_one(
            {"conversationId": conversation_id},
            {
                "$push": {"messages": message.__dict__},
                "$setOnInsert": {"createdAt": datetime.now(tz=UTC)},
            },
            upsert=True,
        )

    async def add_token_usage(self, conversation_id: UUID, token_usage: StageTokenUsage) -> None:
        await self.conversation_history.update_one(
            {"conversationId": conversation_id},
            {
                "$push": {"tokenUsage": {
                    "model": token_usage.model,
                    "stageName": token_usage.stage_name,
                    "inputTokens": token_usage.input_tokens,
                    "outputTokens": token_usage.output_tokens,
                    "timestamp": token_usage.timestamp
                }},
                "$setOnInsert": {"createdAt": datetime.now(tz=UTC)},
            },
            upsert=True,
        )

    async def reset_token_usage(self, conversation_id: UUID) -> None:
        await self.conversation_history.update_one(
            {"conversationId": conversation_id},
            {
                "$set": {"tokenUsage": []}
            }
        )

    async def get_history(self, conversation_id: UUID) -> ConversationHistory | None:
        doc = await self.conversation_history.find_one({"conversationId": conversation_id})

        if doc:
            messages = [
                ChatMessage(
                    role=message["role"],
                    content=message["content"]
                ) for message in doc.get("messages", [])
            ]
            
            token_usage = [
                StageTokenUsage(
                    stage_name=usage["stageName"],
                    model=usage["model"],
                    input_tokens=usage["inputTokens"],
                    output_tokens=usage["outputTokens"],
                    timestamp=usage["timestamp"]
                ) for usage in doc.get("tokenUsage", [])
            ]

            return ConversationHistory(
                conversation_id=conversation_id,
                messages=messages,
                token_usage=token_usage
            )

        return None

    async def create_conversation(self, conversation_id: UUID) -> ConversationHistory:
        await self.conversation_history.insert_one(
            {
                "conversationId": conversation_id,
                "messages": [],
                "createdAt": datetime.now(tz=UTC)
            }
        )

        return ConversationHistory(conversation_id=conversation_id, messages=[], token_usage=[])
