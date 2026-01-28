from unittest.mock import AsyncMock, MagicMock

import pytest

from app.chat import models, service


class DummyModelInfo:
    def __init__(self, name="Model", model_id="mid"):
        self.name = name
        self.model_id = model_id


@pytest.mark.asyncio
async def test_execute_chat_adds_messages_and_saves():
    # Prepare mocks
    chat_agent = AsyncMock()
    # Return one assistant message
    assistant_msg = models.AssistantMessage(
        content="resp",
        model_id="mid",
        model_name="Model",
        usage=models.TokenUsage(1, 2, 3),
    )
    chat_agent.execute_flow.return_value = [assistant_msg]

    conversation_repository = AsyncMock()
    # get returns a conversation
    conversation_repository.get.return_value = models.Conversation()

    model_resolution_service = MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo(
        name="Model", model_id="mid"
    )

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
    )

    conv = await svc.execute_chat("q", "mid")

    # ensure conversation saved and contains assistant message
    conversation_repository.save.assert_awaited()
    assert any(isinstance(m, models.AssistantMessage) for m in conv.messages)
