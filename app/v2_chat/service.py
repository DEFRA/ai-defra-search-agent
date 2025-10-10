from app.v2_chat.agent import AbstractChatAgent


class ChatService:
    def __init__(self, orchestrator: AbstractChatAgent):
        self.orchestrator = orchestrator
    
    async def execute_chat(self, question: str) -> str:
        return await self.orchestrator.execute_flow(question)
