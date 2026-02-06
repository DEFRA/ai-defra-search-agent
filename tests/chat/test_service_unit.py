import json
import uuid

import pytest
from pytest_mock import MockerFixture

from app.chat import models, service
from app.models import UnsupportedModelError


class DummyModelInfo:
    def __init__(self, name="Model", model_id="mid"):
        self.name = name
        self.model_id = model_id


@pytest.mark.asyncio
async def test_execute_chat_adds_messages_and_saves(mocker: MockerFixture):
    # Prepare mocks
    chat_agent = mocker.AsyncMock()
    # Return one assistant message
    assistant_msg = models.AssistantMessage(
        content="resp",
        model_id="mid",
        model_name="Model",
        usage=models.TokenUsage(1, 2, 3),
    )
    chat_agent.execute_flow.return_value = [assistant_msg]

    conversation_repository = mocker.AsyncMock()
    # get returns a conversation
    conversation_repository.get.return_value = models.Conversation()

    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo(
        name="Model", model_id="mid"
    )

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
    )

    conv = await svc.execute_chat("q", "mid", uuid.uuid4())

    conversation_repository.save.assert_awaited()
    assert any(isinstance(m, models.AssistantMessage) for m in conv.messages)


@pytest.mark.asyncio
async def test_build_knowledge_reference_str_and_execute_chat_happy_path(
    mocker: MockerFixture,
):
    # Prepare model resolution service (synchronous resolver)
    model_resolution = mocker.MagicMock()
    model_resolution.resolve_model.return_value = DummyModelInfo(
        name="TestModel", model_id="m1"
    )

    # Prepare conversation repository
    conversation = models.Conversation()
    conversation_repository = mocker.AsyncMock()
    conversation_repository.get = mocker.AsyncMock(return_value=conversation)
    conversation_repository.save = mocker.AsyncMock()

    # Prepare agent responses
    src = models.Source(name="Doc", location="http://x", snippet="line1", score=0.75)
    assistant_msg = models.AssistantMessage(
        content="resp",
        model_id="m1",
        model_name="TestModel",
        usage=models.TokenUsage(1, 2, 3),
        sources=[src],
    )

    chat_agent = mocker.AsyncMock()
    chat_agent.execute_flow = mocker.AsyncMock(return_value=[assistant_msg])

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution,
    )

    result = await svc.execute_chat(
        question="q", model_id="m1", message_id=uuid.uuid4(), conversation_id=None
    )

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


@pytest.mark.asyncio
async def test_queue_chat_creates_new_conversation(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo(
        name="TestModel", model_id="m1"
    )
    sqs_client = mocker.MagicMock()
    sqs_client.__enter__.return_value = sqs_client
    sqs_client.__exit__.return_value = None

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
    )

    message_id, conversation_id, status = await svc.queue_chat(
        question="Hello", model_id="m1"
    )

    assert message_id is not None
    assert conversation_id is not None
    assert status == models.MessageStatus.QUEUED
    conversation_repository.save.assert_awaited_once()
    sqs_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_queue_chat_adds_to_existing_conversation(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    existing_conversation = models.Conversation()
    conversation_repository = mocker.AsyncMock()
    conversation_repository.get.return_value = existing_conversation
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo()
    sqs_client = mocker.MagicMock()
    sqs_client.__enter__.return_value = sqs_client
    sqs_client.__exit__.return_value = None

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
    )

    conv_id = uuid.uuid4()
    message_id, conversation_id, status = await svc.queue_chat(
        question="Hi", model_id="mid", conversation_id=conv_id
    )

    assert conversation_id == existing_conversation.id
    assert len(existing_conversation.messages) == 1
    conversation_repository.get.assert_awaited_once_with(conv_id)
    conversation_repository.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_chat_raises_when_conversation_not_found(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    conversation_repository.get.return_value = None
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo()

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
    )

    with pytest.raises(models.ConversationNotFoundError):
        await svc.queue_chat(
            question="Hi", model_id="mid", conversation_id=uuid.uuid4()
        )


@pytest.mark.asyncio
async def test_queue_chat_raises_when_model_unsupported(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.side_effect = UnsupportedModelError(
        "bad model"
    )

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
    )

    with pytest.raises(UnsupportedModelError):
        await svc.queue_chat(question="Hi", model_id="invalid")


@pytest.mark.asyncio
async def test_queue_chat_sends_correct_sqs_message(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo()
    sqs_client = mocker.MagicMock()
    sqs_client.__enter__.return_value = sqs_client
    sqs_client.__exit__.return_value = None

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
    )

    message_id, conversation_id, _ = await svc.queue_chat(
        question="Test question", model_id="m1"
    )

    sqs_client.send_message.assert_called_once()
    call_args = sqs_client.send_message.call_args[0][0]
    message_data = json.loads(call_args)
    assert message_data["message_id"] == str(message_id)
    assert message_data["conversation_id"] == str(conversation_id)
    assert message_data["question"] == "Test question"
    assert message_data["model_id"] == "m1"


@pytest.mark.asyncio
async def test_queue_chat_without_sqs_client(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo()

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=None,
    )

    message_id, conversation_id, status = await svc.queue_chat(
        question="Hi", model_id="mid"
    )

    assert message_id is not None
    assert conversation_id is not None
    assert status == models.MessageStatus.QUEUED
    conversation_repository.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_chat_handles_sqs_error_gracefully(mocker: MockerFixture):
    chat_agent = mocker.AsyncMock()
    conversation_repository = mocker.AsyncMock()
    model_resolution_service = mocker.MagicMock()
    model_resolution_service.resolve_model.return_value = DummyModelInfo()
    sqs_client = mocker.MagicMock()
    sqs_client.__enter__.return_value = sqs_client
    sqs_client.__exit__.return_value = None
    sqs_client.send_message.side_effect = Exception("SQS error")

    mock_logger = mocker.patch("app.chat.service.logger")

    svc = service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
    )

    message_id, _, _ = await svc.queue_chat(question="Hi", model_id="mid")

    mock_logger.error.assert_called_once()
    assert "Failed to queue message" in str(mock_logger.error.call_args)


@pytest.mark.asyncio
async def test_get_conversation_returns_conversation(mocker: MockerFixture):
    conversation = models.Conversation()
    conversation_repository = mocker.AsyncMock()
    conversation_repository.get.return_value = conversation

    svc = service.ChatService(
        chat_agent=None,
        conversation_repository=conversation_repository,
        model_resolution_service=None,
    )

    conv_id = uuid.uuid4()
    result = await svc.get_conversation(conv_id)

    assert result == conversation
    conversation_repository.get.assert_awaited_once_with(conv_id)


@pytest.mark.asyncio
async def test_get_conversation_raises_error_when_not_found(mocker: MockerFixture):
    conversation_repository = mocker.AsyncMock()
    conversation_repository.get.return_value = None

    svc = service.ChatService(
        chat_agent=None,
        conversation_repository=conversation_repository,
        model_resolution_service=None,
    )

    with pytest.raises(models.ConversationNotFoundError):
        await svc.get_conversation(uuid.uuid4())
