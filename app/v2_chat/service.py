from uuid import UUID
from app.v2_chat.agent import AbstractChatAgent

from app.conversation_history.models import ChatMessage
from app.v2_chat.state_models import ChatState
from app.conversation_history.service import ConversationHistoryService

class ChatService:
    def __init__(
            self,
            orchestrator: AbstractChatAgent,
            history_service: ConversationHistoryService
        ):
        self.orchestrator = orchestrator
        self.history_service = history_service

    async def _setup_chat(self, conversation_id: UUID = None):
        if conversation_id is None:
            return await self.history_service.create_conversation()
        
        return await self.history_service.get_history(conversation_id)

    async def execute_chat(
            self,
            question: str,
            conversation_id: UUID = None
        ) -> tuple[ChatState, UUID]:

        conversation = await self._setup_chat(conversation_id)

        await self.history_service.add_message(
            conversation.conversation_id,
            ChatMessage(role="user", content=question)
        )

        response = await self.orchestrator.execute_flow(question, conversation)

        await self.history_service.add_message(
            conversation.conversation_id,
            ChatMessage(role="assistant", content=response.get("answer"))
        )

        return response, conversation.conversation_id
