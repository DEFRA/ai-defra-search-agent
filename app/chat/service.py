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
            self, question: str, conversation_id: uuid.UUID = None
    ) -> models.Conversation:
        # Get or create conversation from repository
        if conversation_id:
            # If conversation_id provided, it must exist
            conversation = await self.conversation_repository.get(str(conversation_id))
        else:
            # Create new conversation if no ID provided
            conversation = models.Conversation()

        # add message to conversation
        user_message = models.Message(role="user", content=question)
        conversation.add_message(user_message)

        # call chat agent to execute flow with question
        agent_responses = await self.chat_agent.execute_flow(question)

        # handle response - add agent messages to conversation
        for response_message in agent_responses:
            conversation.add_message(response_message)

        # save conversation
        await self.conversation_repository.save(conversation)

        return conversation
