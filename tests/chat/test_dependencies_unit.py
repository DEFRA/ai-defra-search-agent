from app.chat import dependencies
from app.chat import repository as repo_mod
from app.common import sqs as sqs_mod


def test_get_sqs_client_returns_sqs_client():
    # Provide minimal config so SQSClient can be instantiated
    from app import config as app_config

    class Cfg:
        chat_queue_url = "http://example"
        aws_region = "eu-west-2"
        localstack_url = None

    old = app_config.config
    app_config.config = Cfg()
    try:
        c = dependencies.get_sqs_client()
        assert isinstance(c, sqs_mod.SQSClient)
    finally:
        app_config.config = old


def test_get_conversation_repository_with_db():
    # Provide an object with a `conversations` attribute expected by the repo
    class DummyDB:
        pass

    dummy_db = DummyDB()
    dummy_db.conversations = object()
    r = dependencies.get_conversation_repository(dummy_db)  # type: ignore
    assert isinstance(r, repo_mod.MongoConversationRepository)
