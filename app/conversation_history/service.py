import uuid

from app.conversation_history import models, repository
from app.v2_chat import models as v2_chat_models


class ConversationHistoryService:
    """Service class for managing conversation history."""

    def __init__(self, repository: repository.AbstractConversationHistoryRepository):
        self.repository = repository

    async def create_conversation(self) -> models.ConversationHistory:
        conversation_id = uuid.uuid4()

        return await self.repository.create_conversation(conversation_id)

    async def add_message(
        self, conversation_id: uuid.UUID, message: models.ChatMessage
    ) -> None:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.add_message(conversation_id, message)

    async def add_token_usage(
        self, conversation_id: uuid.UUID, usage: v2_chat_models.StageTokenUsage
    ) -> None:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.add_token_usage(conversation_id, usage)

    async def reset_token_usage(self, conversation_id: uuid.UUID) -> None:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        await self.repository.reset_token_usage(conversation_id)

    async def get_history(
        self, conversation_id: uuid.UUID
    ) -> models.ConversationHistory:
        conversation = await self.repository.get_history(conversation_id)

        if conversation is None:
            msg = f"Conversation with ID {conversation_id} not found."
            raise models.ConversationNotFoundError(msg)

        return conversation

    async def _fetch_conversations(
        self, conversation_ids: list[uuid.UUID]
    ) -> list[models.ConversationHistory]:
        conversations = []
        for conversation_id in conversation_ids:
            try:
                conversation = await self.repository.get_history(conversation_id)
                if conversation:
                    conversations.append(conversation)
            except models.ConversationNotFoundError:
                continue
        return conversations

    def _aggregate_token_usage(
        self, conversations: list[models.ConversationHistory]
    ) -> tuple[dict, dict]:
        model_usage = {}
        conversation_usage = {}

        for conversation in conversations:
            conv_id_str = str(conversation.conversation_id)
            conversation_usage[conv_id_str] = self._initialize_usage_entry()

            for token_usage in conversation.token_usage:
                self._update_model_usage(model_usage, token_usage)
                self._update_conversation_usage(
                    conversation_usage[conv_id_str], token_usage
                )

        return model_usage, conversation_usage

    def _initialize_usage_entry(self) -> dict[str, any]:
        return {"total_input_tokens": 0, "total_output_tokens": 0, "models": {}}

    def _update_model_usage(
        self, model_usage: dict, token_usage: v2_chat_models.StageTokenUsage
    ) -> None:
        if token_usage.model not in model_usage:
            model_usage[token_usage.model] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

        model_usage[token_usage.model]["total_input_tokens"] += token_usage.input_tokens
        model_usage[token_usage.model]["total_output_tokens"] += (
            token_usage.output_tokens
        )

    def _update_conversation_usage(
        self, conv_usage: dict, token_usage: v2_chat_models.StageTokenUsage
    ) -> None:
        conv_usage["total_input_tokens"] += token_usage.input_tokens
        conv_usage["total_output_tokens"] += token_usage.output_tokens

        if token_usage.model not in conv_usage["models"]:
            conv_usage["models"][token_usage.model] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

        conv_usage["models"][token_usage.model]["total_input_tokens"] += (
            token_usage.input_tokens
        )
        conv_usage["models"][token_usage.model]["total_output_tokens"] += (
            token_usage.output_tokens
        )

    def _calculate_overall_usage(self, model_usage: dict) -> dict[str, int]:
        total_input = sum(usage["total_input_tokens"] for usage in model_usage.values())
        total_output = sum(
            usage["total_output_tokens"] for usage in model_usage.values()
        )

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    def _format_model_usage(self, model_usage: dict) -> list[dict]:
        return [
            {
                "model": model,
                "total_input_tokens": usage["total_input_tokens"],
                "total_output_tokens": usage["total_output_tokens"],
                "total_tokens": usage["total_input_tokens"]
                + usage["total_output_tokens"],
            }
            for model, usage in model_usage.items()
        ]

    def _format_conversation_usage(self, conversation_usage: dict) -> list[dict]:
        return [
            {
                "conversation_id": conv_id,
                "total_input_tokens": usage["total_input_tokens"],
                "total_output_tokens": usage["total_output_tokens"],
                "total_tokens": usage["total_input_tokens"]
                + usage["total_output_tokens"],
                "models": self._format_model_usage(usage["models"]),
            }
            for conv_id, usage in conversation_usage.items()
        ]

    async def get_bulk_token_usage(
        self, conversation_ids: list[uuid.UUID]
    ) -> dict[str, any]:
        conversations = await self._fetch_conversations(conversation_ids)

        if not conversations:
            return self._get_empty_usage_response()

        model_usage, conversation_usage = self._aggregate_token_usage(conversations)

        return {
            "overall_usage": self._calculate_overall_usage(model_usage),
            "usage_by_model": self._format_model_usage(model_usage),
            "usage_by_conversation": self._format_conversation_usage(
                conversation_usage
            ),
        }

    def _get_empty_usage_response(self) -> dict[str, any]:
        return {
            "overall_usage": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            },
            "usage_by_model": [],
            "usage_by_conversation": [],
        }
