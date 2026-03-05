from pytest_mock import MockerFixture

from app.feedback import dependencies, repository, service

_RETRY_ATTEMPTS = 2
_RETRY_BASE_DELAY_SECONDS = 0.5


def test_get_feedback_repository(mocker: MockerFixture):
    mock_db = mocker.Mock()
    mock_config = mocker.Mock()
    mock_config.mongo.retry_attempts = _RETRY_ATTEMPTS
    mock_config.mongo.retry_base_delay_seconds = _RETRY_BASE_DELAY_SECONDS

    repo = dependencies.get_feedback_repository(db=mock_db, app_config=mock_config)

    assert isinstance(repo, repository.MongoFeedbackRepository)
    assert repo.db == mock_db
    assert repo.retry_attempts == _RETRY_ATTEMPTS
    assert repo.retry_base_delay_seconds == _RETRY_BASE_DELAY_SECONDS


def test_get_feedback_service(mocker: MockerFixture):
    mock_repository = mocker.Mock(spec=repository.AbstractFeedbackRepository)

    feedback_service = dependencies.get_feedback_service(
        feedback_repository=mock_repository
    )

    assert isinstance(feedback_service, service.FeedbackService)
    assert feedback_service.feedback_repository == mock_repository
