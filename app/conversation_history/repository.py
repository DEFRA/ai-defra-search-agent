import abc
import datetime
import uuid

import pymongo.asynchronous.database

from app.conversation_history import models
from app.v2_chat import models as v2_chat_models


class AbstractConversationHistoryRepository(abc.ABC):
    @abc.abstractmethod
    async def add_message(self, conversation_id: uuid.UUID, message: models.ChatMessage) -> None:
        pass

    @abc.abstractmethod
    async def add_token_usage(self, conversation_id: uuid.UUID, token_usage: v2_chat_models.StageTokenUsage) -> None:
        pass

    @abc.abstractmethod
    async def reset_token_usage(self, conversation_id: uuid.UUID) -> None:
        pass

    @abc.abstractmethod
    async def get_history(self, conversation_id: uuid.UUID) -> models.ConversationHistory | None:
        pass

    @abc.abstractmethod
    async def create_conversation(self, conversation_id: uuid.UUID) -> models.ConversationHistory:
        pass


class MongoConversationHistoryRepository(AbstractConversationHistoryRepository):
    def __init__(self, db: pymongo.asynchronous.database.AsyncDatabase):
        self.db: pymongo.asynchronous.database.AsyncDatabase = db
        self.conversation_history: pymongo.asynchronous.database.AsyncCollection = self.db.conversationHistory

    async def add_message(self, conversation_id: uuid.UUID, message: models.ChatMessage) -> None:
        await self.conversation_history.update_one(
            {
                "conversationId": conversation_id
            },
            {
                "$push": {
                    "messages": {
                        "role": message.role,
                        "content": message.content,
                        "context": [
                            {
                                "content": document.content,
                                "snapshotId": document.snapshot_id,
                                "sourceId": document.source_id,
                                "name": document.name,
                                "location": document.location
                            }
                            for document in message.context
                        ] if message.context else None
                    }
                },
                "$setOnInsert": {"createdAt": datetime.datetime.now(tz=datetime.UTC)},
            },
            upsert=True,
        )

    async def add_token_usage(self, conversation_id: uuid.UUID, token_usage: v2_chat_models.StageTokenUsage) -> None:
        await self.conversation_history.update_one(
            {
                "conversationId": conversation_id
            },
            {
                "$push": {
                    "tokenUsage": {
                        "model": token_usage.model,
                        "stageName": token_usage.stage_name,
                        "inputTokens": token_usage.input_tokens,
                        "outputTokens": token_usage.output_tokens,
                        "timestamp": token_usage.timestamp
                    }
                },
                "$setOnInsert": {"createdAt": datetime.datetime.now(tz=datetime.UTC)},
            },
            upsert=True,
        )

    async def reset_token_usage(self, conversation_id: uuid.UUID) -> None:
        await self.conversation_history.update_one(
            {
                "conversationId": conversation_id
            },
            {
                "$set": {"tokenUsage": []}
            }
        )

    async def get_history(self, conversation_id: uuid.UUID) -> models.ConversationHistory | None:
        doc = await self.conversation_history.find_one({"conversationId": conversation_id})

        if doc:
            messages = [
                models.ChatMessage(
                    role=message["role"],
                    content=message["content"],
                    context=message.get("context", None)
                ) for message in doc.get("messages", [])
            ]

            token_usage = [
                v2_chat_models.StageTokenUsage(
                    stage_name=usage["stageName"],
                    model=usage["model"],
                    input_tokens=usage["inputTokens"],
                    output_tokens=usage["outputTokens"],
                    timestamp=usage["timestamp"]
                ) for usage in doc.get("tokenUsage", [])
            ]

            return models.ConversationHistory(
                conversation_id=conversation_id,
                messages=messages,
                token_usage=token_usage
            )

        return None

    async def create_conversation(self, conversation_id: uuid.UUID) -> models.ConversationHistory:
        await self.conversation_history.insert_one(
            {
                "conversationId": conversation_id,
                "messages": [],
                "createdAt": datetime.datetime.now(tz=datetime.UTC)
            }
        )

        return models.ConversationHistory(conversation_id=conversation_id, messages=[], token_usage=[])
