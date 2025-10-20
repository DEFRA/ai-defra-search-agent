from uuid import UUID, uuid4

from app.conversation_history.models import (
    ChatMessage,
    ConversationHistory,
    ConversationNotFoundError
)

from app.v2_chat.models import StageTokenUsage

from app.conversation_history.repository import AbstractConversationHistoryRepository


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

    async def add_token_usage(self, conversation_id: UUID, usage: StageTokenUsage) -> None:
        """Add token usage for a conversation."""
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise ConversationNotFoundError(msg)

        await self.repository.add_token_usage(conversation_id, usage)

    async def reset_token_usage(self, conversation_id: UUID) -> None:
        """Reset token usage for a conversation."""
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise ConversationNotFoundError(msg)

        await self.repository.reset_token_usage(conversation_id)

    async def get_history(self, conversation_id: UUID) -> ConversationHistory:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise ConversationNotFoundError(msg)

        return conversation
