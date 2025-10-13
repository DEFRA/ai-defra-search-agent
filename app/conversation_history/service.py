from app.conversation_history.models import (
    ChatMessage,
    ConversationHistory,
    ConversationNotFoundError,
)

from app.conversation_history.repository import AbstractConversationHistoryRepository

from uuid import uuid4, UUID


class ConversationHistoryService:
    """Service class for managing conversation history."""
    def __init__(self, repository: AbstractConversationHistoryRepository):
        self.repository = repository

    async def create_conversation(self) -> ConversationHistory:
        conversation_id = uuid4()

        return await self.repository.create_conversation(conversation_id)

    async def add_message(self, conversation_id: UUID, message: ChatMessage) -> None:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise ConversationNotFoundError(msg)

        await self.repository.add_message(conversation_id, message)

    async def get_history(self, conversation_id: UUID) -> ConversationHistory:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise ConversationNotFoundError(msg)

        return conversation
