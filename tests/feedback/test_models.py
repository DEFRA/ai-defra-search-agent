import datetime
import uuid

import pytest

from app.feedback import models


def test_feedback_creates_with_required_fields():
    feedback = models.Feedback(was_helpful="very-useful")

    assert feedback.was_helpful == "very-useful"
    assert isinstance(feedback.id, uuid.UUID)
    assert isinstance(feedback.timestamp, datetime.datetime)
    assert feedback.conversation_id is None
    assert feedback.comment is None


def test_feedback_creates_with_all_fields():
    conversation_id = uuid.uuid4()
    comment = "Very helpful information!"

    feedback = models.Feedback(
        was_helpful="useful", conversation_id=conversation_id, comment=comment
    )

    assert feedback.was_helpful == "useful"
    assert feedback.conversation_id == conversation_id
    assert feedback.comment == comment
    assert isinstance(feedback.id, uuid.UUID)
    assert isinstance(feedback.timestamp, datetime.datetime)


def test_feedback_generates_unique_ids():
    feedback1 = models.Feedback(was_helpful="very-useful")
    feedback2 = models.Feedback(was_helpful="not-useful")

    assert feedback1.id != feedback2.id


def test_feedback_generates_timestamp_in_utc():
    before = datetime.datetime.now(datetime.UTC)
    feedback = models.Feedback(was_helpful="neither")
    after = datetime.datetime.now(datetime.UTC)

    assert before <= feedback.timestamp <= after
    assert feedback.timestamp.tzinfo == datetime.UTC


def test_feedback_is_frozen():
    feedback = models.Feedback(was_helpful="not-at-all-useful")

    with pytest.raises(AttributeError):
        feedback.was_helpful = "useful"


def test_feedback_requires_keyword_arguments():
    with pytest.raises(TypeError):
        models.Feedback("very-useful")


def test_feedback_with_not_useful():
    feedback = models.Feedback(was_helpful="not-useful")

    assert feedback.was_helpful == "not-useful"


def test_feedback_with_empty_comment():
    feedback = models.Feedback(was_helpful="useful", comment="")

    assert feedback.comment == ""


def test_feedback_with_long_comment():
    long_comment = "x" * 1200
    feedback = models.Feedback(was_helpful="very-useful", comment=long_comment)

    assert feedback.comment == long_comment
    assert len(feedback.comment) == 1200
