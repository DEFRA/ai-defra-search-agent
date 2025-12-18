from pytest_mock import MockerFixture

from app.feedback import dependencies, repository, service


def test_get_feedback_repository(mocker: MockerFixture):
    mock_db = mocker.Mock()

    repo = dependencies.get_feedback_repository(db=mock_db)

    assert isinstance(repo, repository.MongoFeedbackRepository)
    assert repo.db == mock_db


def test_get_feedback_service(mocker: MockerFixture):
    mock_repository = mocker.Mock(spec=repository.AbstractFeedbackRepository)

    feedback_service = dependencies.get_feedback_service(
        feedback_repository=mock_repository
    )

    assert isinstance(feedback_service, service.FeedbackService)
    assert feedback_service.feedback_repository == mock_repository
