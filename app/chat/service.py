import uuid

from app.chat import agent, models, repository


class ChatService:
    def __init__(
        self,
        chat_agent: agent.AbstractChatAgent,
        conversation_repository: repository.AbstractConversationRepository,
    ):
        self.chat_agent = chat_agent
        self.conversation_repository = conversation_repository

    async def execute_chat(
        self, question: str, model_name: str, conversation_id: uuid.UUID | None = None
    ) -> models.Conversation:
        if conversation_id:
            conversation = await self.conversation_repository.get(conversation_id)
        else:
            conversation = models.Conversation()

        if not conversation:
            msg = f"Conversation with id {conversation_id} not found"
            raise models.ConversationNotFoundError(msg)

        user_message = models.Message(
            role="user",
            content=question,
            usage=models.TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        conversation.add_message(user_message)

        agent_responses = await self.chat_agent.execute_flow(
            question=question, model_name=model_name
        )

        for response_message in agent_responses:
            conversation.add_message(response_message)

        await self.conversation_repository.save(conversation)

        return conversation
