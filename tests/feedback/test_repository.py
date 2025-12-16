import datetime
import uuid

import pytest

from app.feedback import models, repository


@pytest.fixture
def mock_db(mocker):
    db = mocker.MagicMock()
    db.feedback = mocker.AsyncMock()
    return db


@pytest.fixture
def mongo_repository(mock_db):
    return repository.MongoFeedbackRepository(mock_db)


@pytest.mark.asyncio
async def test_save_stores_feedback_with_all_fields(mongo_repository, mock_db):
    feedback_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    feedback = models.Feedback(
        id=feedback_id,
        conversation_id=conversation_id,
        was_helpful=True,
        comment="Very helpful!",
        timestamp=timestamp,
    )

    await mongo_repository.save(feedback)

    mock_db.feedback.update_one.assert_called_once()
    call_args = mock_db.feedback.update_one.call_args

    assert call_args[0][0] == {"feedback_id": feedback_id}

    update_doc = call_args[0][1]
    saved_data = update_doc["$set"]

    assert saved_data["feedback_id"] == feedback_id
    assert saved_data["conversation_id"] == conversation_id
    assert saved_data["was_helpful"] is True
    assert saved_data["comment"] == "Very helpful!"
    assert saved_data["timestamp"] == timestamp


@pytest.mark.asyncio
async def test_save_stores_feedback_with_optional_fields_none(
    mongo_repository, mock_db
):
    feedback_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    feedback = models.Feedback(
        id=feedback_id,
        was_helpful=False,
        timestamp=timestamp,
    )

    await mongo_repository.save(feedback)

    call_args = mock_db.feedback.update_one.call_args
    saved_data = call_args[0][1]["$set"]

    assert saved_data["conversation_id"] is None
    assert saved_data["comment"] is None
    assert saved_data["was_helpful"] is False


@pytest.mark.asyncio
async def test_save_uses_upsert(mongo_repository, mock_db):
    feedback = models.Feedback(was_helpful=True)

    await mongo_repository.save(feedback)

    call_args = mock_db.feedback.update_one.call_args
    assert call_args[1]["upsert"] is True


@pytest.mark.asyncio
async def test_get_retrieves_feedback_with_all_fields(mongo_repository, mock_db):
    feedback_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    mock_doc = {
        "feedback_id": feedback_id,
        "conversation_id": conversation_id,
        "was_helpful": True,
        "comment": "Great response!",
        "timestamp": timestamp,
    }
    mock_db.feedback.find_one.return_value = mock_doc

    result = await mongo_repository.get(feedback_id)

    assert result is not None
    assert result.id == feedback_id
    assert result.conversation_id == conversation_id
    assert result.was_helpful is True
    assert result.comment == "Great response!"
    assert result.timestamp == timestamp


@pytest.mark.asyncio
async def test_get_retrieves_feedback_with_optional_fields_missing(
    mongo_repository, mock_db
):
    feedback_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    mock_doc = {
        "feedback_id": feedback_id,
        "was_helpful": False,
        "timestamp": timestamp,
    }
    mock_db.feedback.find_one.return_value = mock_doc

    result = await mongo_repository.get(feedback_id)

    assert result is not None
    assert result.id == feedback_id
    assert result.conversation_id is None
    assert result.comment is None
    assert result.was_helpful is False


@pytest.mark.asyncio
async def test_get_returns_none_when_feedback_not_found(mongo_repository, mock_db):
    feedback_id = uuid.uuid4()
    mock_db.feedback.find_one.return_value = None

    result = await mongo_repository.get(feedback_id)

    assert result is None
    mock_db.feedback.find_one.assert_called_once_with({"feedback_id": feedback_id})


@pytest.mark.asyncio
async def test_get_handles_empty_comment(mongo_repository, mock_db):
    feedback_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    mock_doc = {
        "feedback_id": feedback_id,
        "was_helpful": True,
        "comment": "",
        "timestamp": timestamp,
    }
    mock_db.feedback.find_one.return_value = mock_doc

    result = await mongo_repository.get(feedback_id)

    assert result is not None
    assert result.comment == ""


@pytest.mark.asyncio
async def test_save_and_get_roundtrip(mongo_repository, mock_db):
    feedback_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    original_feedback = models.Feedback(
        id=feedback_id,
        conversation_id=conversation_id,
        was_helpful=True,
        comment="Excellent!",
        timestamp=timestamp,
    )

    await mongo_repository.save(original_feedback)

    save_call = mock_db.feedback.update_one.call_args
    saved_data = save_call[0][1]["$set"]

    mock_db.feedback.find_one.return_value = saved_data

    retrieved_feedback = await mongo_repository.get(feedback_id)

    assert retrieved_feedback is not None
    assert retrieved_feedback.id == original_feedback.id
    assert retrieved_feedback.conversation_id == original_feedback.conversation_id
    assert retrieved_feedback.was_helpful == original_feedback.was_helpful
    assert retrieved_feedback.comment == original_feedback.comment
    assert retrieved_feedback.timestamp == original_feedback.timestamp
