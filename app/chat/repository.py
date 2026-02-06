import abc
import dataclasses
import datetime
import uuid

import pymongo.asynchronous.database

from app.chat import models

MESSAGES_MESSAGE_ID = "messages.message_id"


@dataclasses.dataclass
class MessageDTO:
    role: str
    content: str
    model: str
    model_name: str
    timestamp: datetime.datetime
    message_id: uuid.UUID | None = None
    status: str | None = None
    error_message: str | None = None
    usage: dict[str, int] | None = None

    @classmethod
    def from_domain(cls, domain_message: models.Message) -> "MessageDTO":
        if isinstance(domain_message, models.UserMessage):
            return cls(
                message_id=domain_message.message_id,
                role=domain_message.role,
                content=domain_message.content,
                model=domain_message.model_id,
                model_name=domain_message.model_name,
                timestamp=domain_message.timestamp,
                status=domain_message.status.value,
                error_message=domain_message.error_message,
            )

        return cls(
            message_id=domain_message.message_id,
            role=domain_message.role,
            content=domain_message.content,
            model=domain_message.model_id,
            model_name=domain_message.model_name,
            timestamp=domain_message.timestamp,
            status=models.MessageStatus.COMPLETED.value,
            usage=dataclasses.asdict(domain_message.usage)
            if isinstance(domain_message, models.AssistantMessage)
            else None,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_domain(self) -> models.Message:
        common_args = {
            "role": self.role,
            "content": self.content,
            "model_id": self.model,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
        }

        if self.message_id:
            common_args["message_id"] = self.message_id

        if self.role == "user":
            status = (
                models.MessageStatus(self.status)
                if self.status
                else models.MessageStatus.COMPLETED
            )
            return models.UserMessage(
                status=status,
                error_message=self.error_message,
                **common_args,
            )

        if self.role == "assistant":
            return models.AssistantMessage(
                usage=models.TokenUsage(**self.usage) if self.usage else None,
                **common_args,
            )

        error_msg = f"Unknown role: {self.role}"
        raise ValueError(error_msg)


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
                        MessageDTO.from_domain(domain_message).to_dict()
                        for domain_message in conversation.messages
                    ],
                }
            },
            upsert=True,
        )

    async def get(self, conversation_id: uuid.UUID) -> models.Conversation | None:
        conversation_doc = await self.conversations.find_one(
            {"conversation_id": conversation_id}
        )

        if not conversation_doc:
            return None

        domain_messages = [
            MessageDTO(**message_data).to_domain()
            for message_data in conversation_doc["messages"]
        ]

        return models.Conversation(
            id=conversation_doc["conversation_id"],
            messages=domain_messages,
        )

    async def update_message_status(
        self,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        status: models.MessageStatus,
        error_message: str | None = None,
    ) -> None:
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
        conversation_doc = await self.conversations.find_one(
            {"conversation_id": conversation_id, MESSAGES_MESSAGE_ID: message_id},
            {"messages.$": 1},
        )
        if not conversation_doc:
            return None
        matched_messages = conversation_doc.get("messages") or []
        if not matched_messages:
            return None
        status_value = matched_messages[0].get("status")
        if not status_value:
            return models.MessageStatus.COMPLETED
        return models.MessageStatus(status_value)
