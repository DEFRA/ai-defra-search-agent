import uuid

from app.chat import agent, models, repository

class ConversationHistoryService:
    def __init__(
        self,
        conversation_repository: repository.AbstractConversationRepository,
    ):
        self.conversation_repository = conversation_repository

    async def save_conversation(self, conversation: models.Conversation) -> None:
        await self.conversation_repository.save(conversation)

    async def get_conversation(self, conversation_id: uuid.UUID) -> models.Conversation:
        conversation = await self.conversation_repository.get(conversation_id)

        if not conversation:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        return conversation


class ChatService:
    def __init__(
        self,
        chat_agent: agent.AbstractChatAgent,
        conversation_repository: repository.AbstractConversationRepository,
        history_service: ConversationHistoryService,
    ):
        self.chat_agent = chat_agent
        self.conversation_repository = conversation_repository
        self.history_service = history_service

    async def execute_chat(
        self, question: str, conversation_id: uuid.UUID = None
    ) -> models.Conversation:
        if conversation_id:
            conversation = await self.history_service.get_conversation(conversation_id)
        else:
            conversation = models.Conversation()

        conversation.add_message(
            message=models.Message(role="user", content=question)
        )

        messages = await self.chat_agent.execute_flow(conversation)

        conversation.add_message(message=messages[-1])

        await self.history_service.save_conversation(conversation)

        return conversation
