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

        user_message = models.UserMessage(
            content=question,
            model_id=model_name,  # TODO: this should be model id but we don't have it yet
        )
        conversation.add_message(user_message)

        # TODO: maybe execute_flow should return both question and response so we can add
        # token count and model-id to the user message?
        agent_responses = await self.chat_agent.execute_flow(
            question=question, model_name=model_name
        )

        for response_message in agent_responses:
            conversation.add_message(response_message)

        await self.conversation_repository.save(conversation)

        return conversation
