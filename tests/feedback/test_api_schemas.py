import datetime
import uuid

import pydantic
import pytest

from app.feedback import api_schemas


def test_feedback_request_with_all_fields():
    conversation_id = uuid.uuid4()
    data = {
        "conversationId": str(conversation_id),
        "wasHelpful": True,
        "comment": "This was very helpful!",
    }

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id == conversation_id
    assert request.was_helpful is True
    assert request.comment == "This was very helpful!"


def test_feedback_request_minimal():
    data = {"wasHelpful": False}

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id is None
    assert request.was_helpful is False
    assert request.comment is None


def test_feedback_request_with_snake_case():
    conversation_id = uuid.uuid4()
    data = {
        "conversation_id": str(conversation_id),
        "was_helpful": True,
        "comment": "Great response!",
    }

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id == conversation_id
    assert request.was_helpful is True
    assert request.comment == "Great response!"


def test_feedback_request_comment_max_length():
    long_comment = "x" * 1001

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(was_helpful=True, comment=long_comment)

    errors = exc_info.value.errors()
    assert any(error["type"] == "string_too_long" for error in errors)


def test_feedback_request_missing_required_field():
    data = {"conversationId": str(uuid.uuid4())}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(**data)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("wasHelpful",) for error in errors)


def test_feedback_request_invalid_uuid():
    data = {"conversationId": "not-a-uuid", "wasHelpful": True}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(**data)

    errors = exc_info.value.errors()
    assert any(error["type"] == "uuid_parsing" for error in errors)


def test_feedback_response_serialization():
    feedback_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    response = api_schemas.FeedbackResponse(
        feedback_id=feedback_id,
        timestamp=timestamp,
    )

    serialized = response.model_dump(mode="json", by_alias=True)

    assert serialized["feedbackId"] == str(feedback_id)
    assert "timestamp" in serialized


def test_feedback_response_with_snake_case():
    feedback_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.UTC)

    response = api_schemas.FeedbackResponse(
        feedback_id=feedback_id,
        timestamp=timestamp,
    )

    assert response.feedback_id == feedback_id
    assert response.timestamp == timestamp


def test_feedback_request_valid_comment_length():
    comment = "x" * 1000  # Exactly at max length

    request = api_schemas.FeedbackRequest(was_helpful=True, comment=comment)

    assert request.comment == comment
    assert len(request.comment) == 1000
