import uuid

from app.conversation_history import models as conv_models
from app.conversation_history import service as conv_service
from app.v2_chat import agent, state_models


class ChatService:
    def __init__(
            self,
            orchestrator: agent.AbstractChatAgent,
            history_service: conv_service.ConversationHistoryService
        ):
        self.orchestrator = orchestrator
        self.history_service = history_service

    async def _setup_chat(self, conversation_id: uuid.UUID = None):
        if conversation_id is None:
            return await self.history_service.create_conversation()

        return await self.history_service.get_history(conversation_id)

    async def execute_chat(
            self,
            question: str,
            conversation_id: uuid.UUID = None
        ) -> tuple[state_models.ChatState, uuid.UUID]:

        conversation = await self._setup_chat(conversation_id)

        await self.history_service.add_message(
            conversation.conversation_id,
            conv_models.ChatMessage(role="user", content=question)
        )

        response = await self.orchestrator.execute_flow(question, conversation)

        await self.history_service.add_message(
            conversation.conversation_id,
            conv_models.ChatMessage(
                role="assistant",
                content=response.get("answer"),
                context=response.get("context")
            )
        )

        await self.history_service.reset_token_usage(conversation.conversation_id)

        for usage in response["token_usage"]:
            await self.history_service.add_token_usage(conversation.conversation_id, usage)

        return response, conversation.conversation_id
