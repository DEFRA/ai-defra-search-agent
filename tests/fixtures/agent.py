from app.chat import agent, models


class StubChatAgent(agent.AbstractChatAgent):
    async def execute_flow(
        self, question: str, conversation: models.Conversation
    ) -> list[models.Message]:
        copy_of_messages = conversation.messages.copy()

        return copy_of_messages + [
            models.Message(
                role="user",
                content=question,
            ),
            models.Message(
                role="assistant",
                content="This is a stub response.",
                model="geni-ai-3.5",
            ),
        ]
