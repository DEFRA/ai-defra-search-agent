import abc
import dataclasses
import uuid

import pymongo.asynchronous.database

from app.chat import models


class AbstractConversationRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, conversation: models.Conversation) -> None:
        """Save the conversation to the repository."""

    @abc.abstractmethod
    async def get(self, conversation_id: uuid.UUID) -> models.Conversation | None:
        """Get the conversation from the repository."""


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
                            "usage": dataclasses.asdict(msg.usage)
                            if isinstance(msg, models.AssistantMessage)
                            else None,
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

        messages: list[models.Message] = []
        for msg in conversation["messages"]:
            role = msg["role"]
            content = msg["content"]
            model_id = msg["model"]

            if role == "user":
                messages.append(
                    models.UserMessage(
                        role=role,
                        content=content,
                        model_id=model_id,
                    )
                )
            elif role == "assistant":
                usage = models.TokenUsage(**msg["usage"])
                messages.append(
                    models.AssistantMessage(
                        role=role,
                        content=content,
                        model_id=model_id,
                        usage=usage,
                    )
                )
            else:
                msg = f"Unknown role: {role}"
                raise ValueError(msg)

        return models.Conversation(
            id=conversation["conversation_id"],
            messages=messages,
        )
