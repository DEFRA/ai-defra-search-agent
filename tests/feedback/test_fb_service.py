import uuid

import pytest

from app.feedback import models, repository, service


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
    # Setup
    conversation_id = uuid.uuid4()
    was_helpful = True
    comment = "This was very helpful!"

    # Execute
    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        conversation_id=conversation_id,
        comment=comment,
    )

    # Assert
    assert isinstance(result, models.Feedback)
    assert result.conversation_id == conversation_id
    assert result.was_helpful is True
    assert result.comment == comment
    assert result.id is not None
    assert result.timestamp is not None

    # Verify repository.save was called once with the feedback
    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.conversation_id == conversation_id
    assert saved_feedback.was_helpful is True
    assert saved_feedback.comment == comment


@pytest.mark.asyncio
async def test_submit_feedback_without_conversation_id(
    feedback_service, mock_repository
):
    # Setup
    was_helpful = False
    comment = "Not helpful"

    # Execute
    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        comment=comment,
    )

    # Assert
    assert isinstance(result, models.Feedback)
    assert result.conversation_id is None
    assert result.was_helpful is False
    assert result.comment == comment
    assert result.id is not None
    assert result.timestamp is not None

    # Verify repository.save was called
    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.conversation_id is None
    assert saved_feedback.was_helpful is False


@pytest.mark.asyncio
async def test_submit_feedback_without_comment(feedback_service, mock_repository):
    # Setup
    conversation_id = uuid.uuid4()
    was_helpful = True

    # Execute
    result = await feedback_service.submit_feedback(
        was_helpful=was_helpful,
        conversation_id=conversation_id,
    )

    # Assert
    assert isinstance(result, models.Feedback)
    assert result.conversation_id == conversation_id
    assert result.was_helpful is True
    assert result.comment is None
    assert result.id is not None
    assert result.timestamp is not None

    # Verify repository.save was called
    mock_repository.save.assert_called_once()
    saved_feedback = mock_repository.save.call_args[0][0]
    assert saved_feedback.comment is None


@pytest.mark.asyncio
async def test_submit_feedback_minimal(feedback_service, mock_repository):
    # Setup - only required field
    was_helpful = False

    # Execute
    result = await feedback_service.submit_feedback(was_helpful=was_helpful)

    # Assert
    assert isinstance(result, models.Feedback)
    assert result.conversation_id is None
    assert result.was_helpful is False
    assert result.comment is None
    assert result.id is not None
    assert result.timestamp is not None

    # Verify repository.save was called
    mock_repository.save.assert_called_once()
