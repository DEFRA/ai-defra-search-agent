import dataclasses
import uuid

from app.chat import agent, models, repository
from app.models import service as model_service


class ChatService:
    def __init__(
        self,
        chat_agent: agent.AbstractChatAgent,
        conversation_repository: repository.AbstractConversationRepository,
        model_resolution_service: model_service.AbstractModelResolutionService,
    ):
        self.chat_agent = chat_agent
        self.conversation_repository = conversation_repository
        self.model_resolution_service = model_resolution_service

    async def execute_chat(
        self, question: str, model_id: str, conversation_id: uuid.UUID | None = None
    ) -> models.Conversation:
        # Resolve model information to get the model name
        model_info = self.model_resolution_service.resolve_model(model_id)

        if conversation_id:
            conversation = await self.conversation_repository.get(conversation_id)
        else:
            conversation = models.Conversation()

        if not conversation:
            msg = f"Conversation with id {conversation_id} not found"
            raise models.ConversationNotFoundError(msg)

        # Only add user message if it doesn't already exist (for async flow, message is pre-created)
        # Check if the last message is a user message with matching content
        should_add_user_message = True
        if conversation.messages:
            last_message = conversation.messages[-1]
            if (
                isinstance(last_message, models.UserMessage)
                and last_message.content == question
                and last_message.status in [models.MessageStatus.QUEUED, models.MessageStatus.PROCESSING]
            ):
                should_add_user_message = False

        if should_add_user_message:
            user_message = models.UserMessage(
                content=question,
                model_id=model_id,
                model_name=model_info.name,
            )
            conversation.add_message(user_message)

        # TODO: maybe execute_flow should return both question and response so we can add
        # token count and model-id to the user message?
        agent_request = models.AgentRequest(
            question=question,
            model_id=model_id,
            conversation=conversation.messages[:-1],
        )
        agent_responses = await self.chat_agent.execute_flow(agent_request)

        for response_message in agent_responses:
            if response_message.sources:
                knowledge_reference_str = self._build_knowledge_reference_str(
                    response_message.sources
                )
                response_message = dataclasses.replace(
                    response_message,
                    content=f"{response_message.content}\n\n{knowledge_reference_str}",
                )

            message_with_model_name = dataclasses.replace(
                response_message,
                model_name=model_info.name,
                model_id=model_info.model_id,
            )
            conversation.add_message(message_with_model_name)

        await self.conversation_repository.save(conversation)

        return conversation

    # TODO: This should not be concerned with markdown formatting, this is a display concern
    def _build_knowledge_reference_str(self, sources: list[models.Source]) -> str:
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            snippet = source.snippet.replace("\n", "\n   > ")
            formatted_sources.append(
                f"{i}. **[{source.name}]({source.location})** ({int(source.score * 100)}%)\n   > {snippet}"
            )

        return "\n\n### Sources\n\n" + "\n\n".join(formatted_sources)
