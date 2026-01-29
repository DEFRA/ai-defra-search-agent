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


@pytest.mark.asyncio
async def test_build_knowledge_reference_str_and_execute_chat_happy_path():
    # Prepare model resolution service (synchronous resolver)
    model_resolution = MagicMock()
    model_resolution.resolve_model.return_value = DummyModelInfo(
        name="TestModel", model_id="m1"
    )

    # Prepare conversation repository
    conversation = models.Conversation()
    conversation_repository = AsyncMock()
    conversation_repository.get = AsyncMock(return_value=conversation)
    conversation_repository.save = AsyncMock()

    # Prepare agent responses
    src = models.Source(name="Doc", location="http://x", snippet="line1", score=0.75)
    assistant_msg = models.AssistantMessage(
        content="resp",
        model_id="m1",
        model_name="TestModel",
        usage=models.TokenUsage(1, 2, 3),
        sources=[src],
    )

    chat_agent = AsyncMock()
    chat_agent.execute_flow = AsyncMock(return_value=[assistant_msg])

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution,
    )

    # When conversation_id is provided, repo.get called
    result = await svc.execute_chat(question="q", model_id="m1", conversation_id=None)

    # Ensure messages were added and saved
    assert any(isinstance(m, models.AssistantMessage) for m in result.messages)
    conversation_repository.save.assert_awaited_once()


def test_build_knowledge_reference_str_formatting():
    svc = service.ChatService(None, None, None)
    src1 = models.Source(name="A", location="http://a", snippet="s1\ns2", score=0.5)
    src2 = models.Source(name="B", location="http://b", snippet="s3", score=0.8)
    out = svc._build_knowledge_reference_str([src1, src2])
    assert "### Sources" in out
    assert "[A](http://a)" in out
    assert "[B](http://b)" in out
