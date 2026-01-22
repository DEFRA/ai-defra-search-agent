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

    @abc.abstractmethod
    async def update_message_status(
        self,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        status: models.MessageStatus,
        error_message: str | None = None,
        error_code: int | None = None,
    ) -> None:
        """Update the status of a specific message."""


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
                            "message_id": msg.message_id,
                            "role": msg.role,
                            "content": msg.content,
                            "model": msg.model_id,
                            "model_name": msg.model_name,
                            "status": msg.status.value,
                            "error_message": msg.error_message,
                            "error_code": msg.error_code,
                            "timestamp": msg.timestamp,
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
            model_name = msg["model_name"]
            timestamp = msg["timestamp"]
            message_id = msg.get("message_id")
            status_str = msg.get("status", models.MessageStatus.COMPLETED.value)
            error_message = msg.get("error_message")
            error_code = msg.get("error_code")

            common_args = {
                "role": role,
                "content": content,
                "model_id": model_id,
                "model_name": model_name,
                "timestamp": timestamp,
                "status": models.MessageStatus(status_str),
                "error_message": error_message,
                "error_code": error_code,
            }

            if message_id:
                common_args["message_id"] = message_id

            if role == "user":
                messages.append(models.UserMessage(**common_args))
            elif role == "assistant":
                usage = models.TokenUsage(**msg["usage"])
                messages.append(
                    models.AssistantMessage(
                        usage=usage,
                        **common_args,
                    )
                )
            else:
                msg = f"Unknown role: {role}"
                raise ValueError(msg)

        return models.Conversation(
            id=conversation["conversation_id"],
            messages=messages,
        )

    async def update_message_status(
        self,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        status: models.MessageStatus,
        error_message: str | None = None,
        error_code: int | None = None,
    ) -> None:
        """Update the status of a specific message in a conversation."""
        update_data = {
            "messages.$.status": status.value,
        }
        if error_message is not None:
            update_data["messages.$.error_message"] = error_message
        if error_code is not None:
            update_data["messages.$.error_code"] = error_code

        await self.conversations.update_one(
            {"conversation_id": conversation_id, "messages.message_id": message_id},
            {"$set": update_data},
        )
