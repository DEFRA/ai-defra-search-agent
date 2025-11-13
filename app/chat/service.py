import uuid

from app.chat import agent


class ChatService:
    def __init__(self, chat_agent: agent.AbstractChatAgent):
        self.orchestrator = chat_agent

    async def execute_chat(
        self, question: str, conversation_id: uuid.UUID = None
    ) -> tuple:
        pass
