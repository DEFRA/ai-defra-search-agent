import abc
import dataclasses
import uuid

import pymongo.asynchronous.database

from app.chat import models


class AbstractConversationRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, conversation: models.Conversation) -> None:
        pass

    @abc.abstractmethod
    async def get(self, conversation_id: uuid.UUID) -> models.Conversation | None:
        pass


class MongoConversationRepository(AbstractConversationRepository):
    def __init__(self, db: pymongo.asynchronous.database.AsyncDatabase):
        self.db = db
        self.conversations: pymongo.asynchronous.collection.AsyncCollection = (
            self.db.conversations
        )

    async def save(self, conversation: models.Conversation) -> None:
        await self.conversations.update_one(
            {"conversation_id": conversation.id},
            {
                "$set": {
                    "conversation_id": conversation.id,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "model": msg.model_id,
                            "usage": dataclasses.asdict(msg.usage),
                        }
                        for msg in conversation.messages
                    ],
                }
            },
            upsert=True,
        )

    async def get(self, conversation_id: uuid.UUID) -> models.Conversation | None:
        conversation = await self.conversations.find_one(
            {"conversation_id": conversation_id}
        )

        if not conversation:
            return None

        return models.Conversation(
            id=conversation["conversation_id"],
            messages=[
                models.Message(
                    role=msg["role"],
                    content=msg["content"],
                    model_id=msg.get("model", None),
                    usage=models.TokenUsage(**msg["usage"]),
                )
                for msg in conversation["messages"]
            ],
        )
