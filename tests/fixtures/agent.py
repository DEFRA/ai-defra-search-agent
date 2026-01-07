from app.chat import agent, models


class StubChatAgent(agent.AbstractChatAgent):
    async def execute_flow(self, request: models.AgentRequest) -> list[models.Message]:
        copy_of_messages = request.conversation.copy() if request.conversation else []

        return copy_of_messages + [
            models.Message(
                role="user",
                content=request.question,
            ),
            models.Message(
                role="assistant",
                content="This is a stub response.",
                model_id=request.model_id,
            ),
        ]
