import abc
import dataclasses
import uuid

import pymongo.asynchronous.database

from app.chat import models

MESSAGES_MESSAGE_ID = "messages.message_id"


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
    ) -> None:
        """Update the status of a specific message."""

    @abc.abstractmethod
    async def claim_message(
        self, conversation_id: uuid.UUID, message_id: uuid.UUID
    ) -> bool:
        """Attempt to reserve a queued message by checking the current status
        and setting it to `PROCESSING` in a single repository update.

        Returns True if the reservation succeeded (status changed from
        `QUEUED` to `PROCESSING`), or False if the message was not in a
        `QUEUED` state.
        """

    @abc.abstractmethod
    async def get_message_status(
        self, conversation_id: uuid.UUID, message_id: uuid.UUID
    ) -> models.MessageStatus | None:
        """Return the current status for a message, or None if not found."""


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
                            "status": msg.status.value
                            if isinstance(msg, models.UserMessage)
                            else models.MessageStatus.COMPLETED.value,
                            "error_message": msg.error_message
                            if isinstance(msg, models.UserMessage)
                            else None,
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
            common_args = {
                "role": role,
                "content": content,
                "model_id": model_id,
                "model_name": model_name,
                "timestamp": timestamp,
            }

            if message_id:
                common_args["message_id"] = message_id

            if role == "user":
                status_str = msg.get("status", models.MessageStatus.COMPLETED.value)
                error_message = msg.get("error_message")
                messages.append(
                    models.UserMessage(
                        status=models.MessageStatus(status_str),
                        error_message=error_message,
                        **common_args,
                    )
                )
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
    ) -> None:
        """Update the status of a specific message in a conversation."""
        update_data = {
            "messages.$.status": status.value,
        }
        if error_message is not None:
            update_data["messages.$.error_message"] = error_message

        await self.conversations.update_one(
            {"conversation_id": conversation_id, MESSAGES_MESSAGE_ID: message_id},
            {"$set": update_data},
        )

    async def claim_message(
        self, conversation_id: uuid.UUID, message_id: uuid.UUID
    ) -> bool:
        """Reserve the message by changing its status from `QUEUED` to `PROCESSING`
        using a single update that checks the current status and sets it.

        Returns True when the update matched a document and modified the
        status. Returns False when no matching queued message was found.
        """
        result = await self.conversations.update_one(
            {
                "conversation_id": conversation_id,
                MESSAGES_MESSAGE_ID: message_id,
                "messages.status": models.MessageStatus.QUEUED.value,
            },
            {
                "$set": {
                    "messages.$.status": models.MessageStatus.PROCESSING.value,
                }
            },
        )
        return getattr(result, "matched_count", 0) == 1

    async def get_message_status(
        self, conversation_id: uuid.UUID, message_id: uuid.UUID
    ) -> models.MessageStatus | None:
        """Retrieve the message status for a specific message.

        Returns a `MessageStatus` or ``None`` when the message or
        conversation could not be found.
        """
        doc = await self.conversations.find_one(
            {"conversation_id": conversation_id, MESSAGES_MESSAGE_ID: message_id},
            {"messages.$": 1},
        )
        if not doc:
            return None
        msgs = doc.get("messages") or []
        if not msgs:
            return None
        status_str = msgs[0].get("status")
        if not status_str:
            return models.MessageStatus.COMPLETED
        return models.MessageStatus(status_str)
