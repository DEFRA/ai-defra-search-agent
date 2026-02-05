import uuid
from unittest.mock import AsyncMock

import pytest

from app.chat import models
from app.chat.repository import MongoConversationRepository


@pytest.mark.asyncio
async def test_get_returns_none_when_conversation_missing():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.find_one = AsyncMock(return_value=None)

    repo = MongoConversationRepository(dummy)

    result = await repo.get(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_message_status_doc_not_found_returns_none():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.find_one = AsyncMock(return_value=None)

    repo = MongoConversationRepository(dummy)

    status = await repo.get_message_status(uuid.uuid4(), uuid.uuid4())
    assert status is None


@pytest.mark.asyncio
async def test_get_message_status_empty_messages_returns_none():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.find_one = AsyncMock(return_value={"messages": []})

    repo = MongoConversationRepository(dummy)

    status = await repo.get_message_status(uuid.uuid4(), uuid.uuid4())
    assert status is None


@pytest.mark.asyncio
async def test_get_message_status_defaults_to_completed_when_missing_status():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.find_one = AsyncMock(
        return_value={"messages": [{"message_id": uuid.uuid4()}]}
    )

    repo = MongoConversationRepository(dummy)

    status = await repo.get_message_status(uuid.uuid4(), uuid.uuid4())
    assert status == models.MessageStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_message_status_returns_explicit_status():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.find_one = AsyncMock(
        return_value={"messages": [{"status": models.MessageStatus.PROCESSING.value}]}
    )

    repo = MongoConversationRepository(dummy)

    status = await repo.get_message_status(uuid.uuid4(), uuid.uuid4())
    assert status == models.MessageStatus.PROCESSING


@pytest.mark.asyncio
async def test_update_message_status_sets_error_fields_when_provided():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.update_one = AsyncMock()

    repo = MongoConversationRepository(dummy)

    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()

    await repo.update_message_status(
        conversation_id=conversation_id,
        message_id=message_id,
        status=models.MessageStatus.FAILED,
        error_message="boom",
    )

    # verify update_one called with $set containing the optional fields
    called = dummy.conversations.update_one.call_args_list[0][0][1]
    set_payload = called.get("$set", {})
    assert set_payload["messages.$.status"] == models.MessageStatus.FAILED.value
    assert set_payload["messages.$.error_message"] == "boom"


@pytest.mark.asyncio
async def test_update_message_status_omits_optional_fields_when_none():
    class DummyDB:
        pass

    dummy = DummyDB()
    dummy.conversations = AsyncMock()
    dummy.conversations.update_one = AsyncMock()

    repo = MongoConversationRepository(dummy)

    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()

    await repo.update_message_status(
        conversation_id=conversation_id,
        message_id=message_id,
        status=models.MessageStatus.COMPLETED,
        error_message=None,
    )

    called = dummy.conversations.update_one.call_args_list[0][0][1]
    set_payload = called.get("$set", {})
    assert set_payload["messages.$.status"] == models.MessageStatus.COMPLETED.value
    assert "messages.$.error_message" not in set_payload
