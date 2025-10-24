import uuid

from app.conversation_history import models, repository
from app.v2_chat import models as v2_chat_models


class ConversationHistoryService:
    """Service class for managing conversation history."""
    def __init__(self, repository: repository.AbstractConversationHistoryRepository):
        self.repository = repository

    async def create_conversation(self) -> models.ConversationHistory:
        conversation_id = uuid.uuid4()

        return await self.repository.create_conversation(conversation_id)

    async def add_message(self, conversation_id: uuid.UUID, message: models.ChatMessage) -> None:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.add_message(conversation_id, message)

    async def add_token_usage(self, conversation_id: uuid.UUID, usage: v2_chat_models.StageTokenUsage) -> None:
        """Add token usage for a conversation."""
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.add_token_usage(conversation_id, usage)

    async def reset_token_usage(self, conversation_id: uuid.UUID) -> None:
        """Reset token usage for a conversation."""
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.reset_token_usage(conversation_id)

    async def get_history(self, conversation_id: uuid.UUID) -> models.ConversationHistory:
        conversation = await self.repository.get_history(conversation_id)

        print(conversation)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        return conversation
