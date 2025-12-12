import dataclasses
import datetime
import uuid

import pytest

from app.chat import models, repository


@pytest.fixture
def mock_db(mocker):
    db = mocker.MagicMock()
    db.conversations = mocker.AsyncMock()
    return db


@pytest.fixture
def mongo_repository(mock_db):
    return repository.MongoConversationRepository(mock_db)


@pytest.mark.asyncio
async def test_save_stores_usage_data(mongo_repository, mock_db):
    conversation_id = uuid.uuid4()
    usage = models.TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    message = models.AssistantMessage(content="Hello", model_id="gpt-4", usage=usage)
    conversation = models.Conversation(id=conversation_id, messages=[message])

    await mongo_repository.save(conversation)

    mock_db.conversations.update_one.assert_called_once()
    call_args = mock_db.conversations.update_one.call_args

    assert call_args[0][0] == {"conversation_id": conversation_id}

    update_doc = call_args[0][1]
    saved_messages = update_doc["$set"]["messages"]
    assert len(saved_messages) == 1
    saved_msg = saved_messages[0]

    assert saved_msg["role"] == "assistant"
    assert saved_msg["content"] == "Hello"
    assert saved_msg["model"] == "gpt-4"
    assert saved_msg["usage"] == dataclasses.asdict(usage)
    assert saved_msg["usage"]["input_tokens"] == 10
    assert saved_msg["usage"]["output_tokens"] == 20
    assert saved_msg["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_save_stores_none_when_usage_missing(mongo_repository, mock_db):
    conversation = models.Conversation(
        id=uuid.uuid4(), messages=[models.UserMessage(content="Hi", model_id="gpt-4")]
    )

    await mongo_repository.save(conversation)

    call_args = mock_db.conversations.update_one.call_args
    saved_msg = call_args[0][1]["$set"]["messages"][0]
    assert saved_msg["usage"] is None


@pytest.mark.asyncio
async def test_get_retrieves_usage_data(mongo_repository, mock_db):
    conversation_id = uuid.uuid4()
    usage_dict = {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}

    mock_doc = {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": "assistant",
                "content": "Response",
                "model": "claude-3",
                "usage": usage_dict,
                "timestamp": datetime.datetime.now(datetime.UTC),
            }
        ],
    }
    mock_db.conversations.find_one.return_value = mock_doc

    result = await mongo_repository.get(conversation_id)

    assert result is not None
    assert result.id == conversation_id
    assert len(result.messages) == 1
    msg = result.messages[0]
    assert isinstance(msg, models.AssistantMessage)
    assert isinstance(msg.usage, models.TokenUsage)
    assert msg.usage.input_tokens == 5
    assert msg.usage.output_tokens == 10
    assert msg.usage.total_tokens == 15
    assert msg.model_id == "claude-3"


@pytest.mark.asyncio
async def test_get_raises_when_unknown_role(mongo_repository, mock_db):
    conversation_id = uuid.uuid4()
    mock_doc = {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": "unknown",
                "content": "Response",
                "model": "gpt-4",
                "timestamp": datetime.datetime.now(datetime.UTC),
            }
        ],
    }
    mock_db.conversations.find_one.return_value = mock_doc
    with pytest.raises(ValueError, match="Unknown role: unknown"):
        await mongo_repository.get(conversation_id)


@pytest.mark.asyncio
async def test_save_stores_timestamp(mongo_repository, mock_db):
    conversation_id = uuid.uuid4()
    timestamp = datetime.datetime(2025, 8, 30, 12, 0, 0, tzinfo=datetime.UTC)
    message = models.UserMessage(content="Hello", model_id="gpt-4", timestamp=timestamp)
    conversation = models.Conversation(id=conversation_id, messages=[message])

    await mongo_repository.save(conversation)

    mock_db.conversations.update_one.assert_called_once()
    call_args = mock_db.conversations.update_one.call_args

    update_doc = call_args[0][1]
    saved_msg = update_doc["$set"]["messages"][0]
    assert saved_msg["timestamp"] == timestamp


@pytest.mark.asyncio
async def test_get_retrieves_timestamp(mongo_repository, mock_db):
    conversation_id = uuid.uuid4()
    timestamp = datetime.datetime(2025, 8, 30, 12, 0, 0, tzinfo=datetime.UTC)

    mock_doc = {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": "user",
                "content": "Hi",
                "model": "gpt-4",
                "timestamp": timestamp,
            }
        ],
    }
    mock_db.conversations.find_one.return_value = mock_doc

    result = await mongo_repository.get(conversation_id)

    assert result is not None
    msg = result.messages[0]
    assert msg.timestamp == timestamp
