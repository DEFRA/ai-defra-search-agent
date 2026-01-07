import datetime
import uuid

import pydantic
import pytest

from app.feedback import api_schemas
from app.feedback.models import WasHelpfulRating


def test_feedback_request_with_all_fields():
    conversation_id = uuid.uuid4()
    data = {
        "conversationId": str(conversation_id),
        "wasHelpful": "very-useful",
        "comment": "This was very helpful!",
    }

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id == conversation_id
    assert request.was_helpful == WasHelpfulRating.VERY_USEFUL
    assert request.comment == "This was very helpful!"


def test_feedback_request_minimal():
    data = {"wasHelpful": "not-useful"}

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id is None
    assert request.was_helpful == WasHelpfulRating.NOT_USEFUL
    assert request.comment is None


def test_feedback_request_with_snake_case():
    conversation_id = uuid.uuid4()
    data = {
        "conversation_id": str(conversation_id),
        "was_helpful": "useful",
        "comment": "Great response!",
    }

    request = api_schemas.FeedbackRequest(**data)

    assert request.conversation_id == conversation_id
    assert request.was_helpful == WasHelpfulRating.USEFUL
    assert request.comment == "Great response!"


def test_feedback_request_comment_max_length():
    long_comment = "x" * 1201

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(was_helpful="neither", comment=long_comment)

    errors = exc_info.value.errors()
    assert any(error["type"] == "string_too_long" for error in errors)


def test_feedback_request_missing_required_field():
    data = {"conversationId": str(uuid.uuid4())}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(**data)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("wasHelpful",) for error in errors)


def test_feedback_request_invalid_uuid():
    data = {"conversationId": "not-a-uuid", "wasHelpful": "very-useful"}

    with pytest.raises(pydantic.ValidationError) as exc_info:
        api_schemas.FeedbackRequest(**data)

    errors = exc_info.value.errors()
    assert any(error["type"] == "uuid_parsing" for error in errors)


def test_feedback_request_all_valid_was_helpful_values():
    """Test all 5 valid was_helpful values are accepted"""
    valid_values = [
        ("very-useful", WasHelpfulRating.VERY_USEFUL),
        ("useful", WasHelpfulRating.USEFUL),
        ("neither", WasHelpfulRating.NEITHER),
        ("not-useful", WasHelpfulRating.NOT_USEFUL),
        ("not-at-all-useful", WasHelpfulRating.NOT_AT_ALL_USEFUL),
    ]

    for string_value, enum_value in valid_values:
        request = api_schemas.FeedbackRequest(was_helpful=string_value)
        assert request.was_helpful == enum_value


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
    comment = "x" * 1200  # Exactly at max length

    request = api_schemas.FeedbackRequest(was_helpful="very-useful", comment=comment)

    assert request.comment == comment
    assert len(request.comment) == 1200
