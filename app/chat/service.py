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
        generated_id = conversation_id or uuid.uuid4()

        # mock conversation for now
        conversation = models.Conversation(
            id=str(generated_id),
            messages=[
                models.Message(
                    role="user",
                    content=question,
                    model=None,
                )
            ],
        )

        messages = await self.chat_agent.execute_flow(conversation)

        conversation.add_message(message=messages[-1])

        return conversation
