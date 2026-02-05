import dataclasses
import json
import logging
import uuid

from app.chat import agent, models, repository
from app.common import sqs
from app.models import service as model_service

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        chat_agent: agent.AbstractChatAgent,
        conversation_repository: repository.AbstractConversationRepository,
        model_resolution_service: model_service.AbstractModelResolutionService,
        sqs_client: sqs.SQSClient | None = None,
    ):
        self.chat_agent = chat_agent
        self.conversation_repository = conversation_repository
        self.model_resolution_service = model_resolution_service
        self.sqs_client = sqs_client

    async def execute_chat(
        self,
        question: str,
        model_id: str,
        message_id: uuid.UUID,
        conversation_id: uuid.UUID | None = None,
    ) -> models.Conversation:
        model_info = self.model_resolution_service.resolve_model(model_id)

        if conversation_id:
            conversation = await self.conversation_repository.get(conversation_id)
        else:
            conversation = models.Conversation()

        if not conversation:
            msg = f"Conversation with id {conversation_id} not found"
            raise models.ConversationNotFoundError(msg)

        message_exists = any(m.message_id == message_id for m in conversation.messages)
        if not message_exists:
            user_message = models.UserMessage(
                message_id=message_id,
                content=question,
                model_id=model_id,
                model_name=model_info.name,
            )
            conversation.add_message(user_message)

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

    def _build_knowledge_reference_str(self, sources: list[models.Source]) -> str:
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            snippet = source.snippet.replace("\n", "\n   > ")
            formatted_sources.append(
                f"{i}. **[{source.name}]({source.location})** ({int(source.score * 100)}%)\n   > {snippet}"
            )

        return "\n\n### Sources\n\n" + "\n\n".join(formatted_sources)

    async def queue_chat(
        self,
        question: str,
        model_id: str,
        conversation_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID, uuid.UUID, models.MessageStatus]:
        """Queue a chat message for async processing via SQS."""
        resolved_model = self.model_resolution_service.resolve_model(model_id)

        user_message = models.UserMessage(
            content=question,
            model_id=model_id,
            model_name=resolved_model.name if resolved_model else model_id,
            status=models.MessageStatus.QUEUED,
        )

        if conversation_id:
            conversation = await self.conversation_repository.get(conversation_id)
            if not conversation:
                msg = f"Conversation with id {conversation_id} not found"
                raise models.ConversationNotFoundError(msg)
            conversation.add_message(user_message)
        else:
            conversation = models.Conversation(messages=[user_message])

        await self.conversation_repository.save(conversation)

        if self.sqs_client:
            try:
                with self.sqs_client:
                    self.sqs_client.send_message(
                        json.dumps(
                            {
                                "message_id": str(user_message.message_id),
                                "conversation_id": str(conversation.id),
                                "question": question,
                                "model_id": model_id,
                            }
                        )
                    )
                logger.info(
                    "Successfully queued message %s to SQS", user_message.message_id
                )
            except Exception as e:
                logger.error(
                    "Failed to queue message %s to SQS: %s", user_message.message_id, e
                )

        return user_message.message_id, conversation.id, user_message.status

    async def get_conversation(self, conversation_id: uuid.UUID) -> models.Conversation:
        """Retrieve a conversation by ID."""
        conversation = await self.conversation_repository.get(conversation_id)
        if not conversation:
            msg = f"Conversation with id {conversation_id} not found"
            raise models.ConversationNotFoundError(msg)
        return conversation
