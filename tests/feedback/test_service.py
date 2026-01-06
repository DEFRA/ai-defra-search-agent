import uuid

import pytest

from app.feedback import models, repository, service
from app.feedback.models import WasHelpfulRating


@pytest.fixture
def mock_repository(mocker):
    """Async mock of AbstractFeedbackRepository"""
    return mocker.AsyncMock(spec=repository.AbstractFeedbackRepository)


@pytest.fixture
def feedback_service(mock_repository):
    """FeedbackService instance with mocked dependencies"""
    return service.FeedbackService(feedback_repository=mock_repository)


@pytest.mark.asyncio
async def test_submit_feedback_with_all_fields(feedback_service, mock_repository):
    conversation_id = uuid.uuid4()
    was_helpful = "useful"
    comment = "This was very helpful!"

    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        conversation_id=conversation_id,
        comment=comment,
    )

    assert isinstance(result, models.Feedback)
    assert result.conversation_id == conversation_id
    assert result.was_helpful == WasHelpfulRating.USEFUL
    assert result.comment == comment
    assert result.id is not None
    assert result.timestamp is not None

    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.conversation_id == conversation_id
    assert saved_feedback.was_helpful == WasHelpfulRating.USEFUL
    assert saved_feedback.comment == comment


@pytest.mark.asyncio
async def test_submit_feedback_without_conversation_id(
    feedback_service, mock_repository
):
    was_helpful = "not-useful"
    comment = "Not helpful"

    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        comment=comment,
    )

    assert isinstance(result, models.Feedback)
    assert result.conversation_id is None
    assert result.was_helpful == WasHelpfulRating.NOT_USEFUL
    assert result.comment == comment
    assert result.id is not None
    assert result.timestamp is not None

    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.conversation_id is None
    assert saved_feedback.was_helpful == WasHelpfulRating.NOT_USEFUL


@pytest.mark.asyncio
async def test_submit_feedback_without_comment(feedback_service, mock_repository):
    conversation_id = uuid.uuid4()
    was_helpful = "very-useful"

    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        conversation_id=conversation_id,
    )

    assert isinstance(result, models.Feedback)
    assert result.conversation_id == conversation_id
    assert result.was_helpful == WasHelpfulRating.VERY_USEFUL
    assert result.comment is None
    assert result.id is not None
    assert result.timestamp is not None

    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.comment is None


@pytest.mark.asyncio
async def test_submit_feedback_minimal(feedback_service, mock_repository):
    was_helpful = "neither"

    result = await feedback_service.submit_feedback(was_helpful=was_helpful)

    assert isinstance(result, models.Feedback)
    assert result.conversation_id is None
    assert result.was_helpful == WasHelpfulRating.NEITHER
    assert result.comment is None
    assert result.id is not None
    assert result.timestamp is not None

    mock_repository.save.assert_called_once()
