import pytest

from app.common import mongo
from app.config import config


# Reset the global client variable before each test
@pytest.fixture(autouse=True)
def reset_mongo_client():
    mongo.client = None
    mongo.db = None
    yield
    mongo.client = None
    mongo.db = None


@pytest.mark.asyncio
async def test_get_mongo_client_initialization(mocker):
    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")
    mock_instance = mock_client_cls.return_value

    # Setup the async ping command
    # get_database() returns a DB object, which has an async command() method
    mock_db = mocker.MagicMock()
    mock_db.conversation_history.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})

    client = await mongo.get_mongo_client()

    assert client == mock_instance
    mock_client_cls.assert_called_once_with(
        config.mongo.uri, uuidRepresentation="standard"
    )
    mock_db.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_get_mongo_client_with_custom_tls(mocker, monkeypatch):
    # Mock config and custom certs
    monkeypatch.setattr(config.mongo, "truststore", "custom-cert-key")
    mocker.patch.dict(
        "app.common.tls.custom_ca_certs", {"custom-cert-key": "/path/to/cert.pem"}
    )

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")
    mock_instance = mock_client_cls.return_value
    mock_db = mocker.MagicMock()
    mock_db.conversation_history.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})

    await mongo.get_mongo_client()

    # Verify TLS param was passed
    mock_client_cls.assert_called_once_with(
        config.mongo.uri, tlsCAFile="/path/to/cert.pem", uuidRepresentation="standard"
    )


@pytest.mark.asyncio
async def test_get_mongo_client_returns_existing(mocker):
    # Set an existing client
    existing_client = mocker.Mock()
    mongo.client = existing_client

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")

    result = await mongo.get_mongo_client()

    # Should return existing without creating new one or pinging
    assert result == existing_client
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_get_db(mocker):
    mock_client = mocker.MagicMock()
    mock_db = mocker.Mock()

    # Setup mock for create_index to be awaitable
    mock_db.conversation_history.create_index = mocker.AsyncMock()

    mock_client.get_database.return_value = mock_db

    # First call initializes
    result = await mongo.get_db(mock_client)
    assert result == mock_db

    # Check first argument (database name)
    assert mock_client.get_database.call_args[0][0] == config.mongo.database

    # Second call returns cached
    result2 = await mongo.get_db(mock_client)
    assert result2 == mock_db
    assert mock_client.get_database.call_count == 1
