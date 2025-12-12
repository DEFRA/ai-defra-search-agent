import pytest

from app.common import mongo


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

    mock_config = mocker.Mock()
    mock_config.mongo.uri = "mongodb://localhost:27017"
    mock_config.mongo.truststore = None

    # Setup the async ping command
    # get_database() returns a DB object, which has an async command() method
    mock_db = mocker.MagicMock()
    mock_db.conversation_history.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})
    mock_instance.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    client = await mongo.get_mongo_client(app_config=mock_config)

    assert client == mock_instance
    mock_client_cls.assert_called_once_with(
        mock_config.mongo.uri, uuidRepresentation="standard"
    )
    mock_instance.admin.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_get_mongo_client_with_custom_tls(mocker):
    mock_config = mocker.Mock()
    mock_config.mongo.uri = "mongodb://localhost:27017"
    mock_config.mongo.truststore = "custom-cert-key"

    mocker.patch.dict(
        "app.common.tls.custom_ca_certs", {"custom-cert-key": "/path/to/cert.pem"}
    )

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")
    mock_instance = mock_client_cls.return_value
    mock_db = mocker.MagicMock()
    mock_db.conversation_history.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})
    mock_instance.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    await mongo.get_mongo_client(app_config=mock_config)

    # Verify TLS param was passed
    mock_client_cls.assert_called_once_with(
        mock_config.mongo.uri,
        tlsCAFile="/path/to/cert.pem",
        uuidRepresentation="standard",
    )


@pytest.mark.asyncio
async def test_get_mongo_client_returns_existing(mocker):
    # Set an existing client
    existing_client = mocker.Mock()
    mongo.client = existing_client
    mock_config = mocker.Mock()

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")

    result = await mongo.get_mongo_client(app_config=mock_config)

    # Should return existing without creating new one or pinging
    assert result == existing_client
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_get_db(mocker):
    mock_client = mocker.MagicMock()
    mock_db = mocker.Mock()
    mock_config = mocker.Mock()
    mock_config.mongo.database = "test_db"

    # Setup mock for create_index to be awaitable
    mock_db.conversation_history.create_index = mocker.AsyncMock()

    mock_client.get_database.return_value = mock_db

    # First call initializes
    result = await mongo.get_db(client=mock_client, app_config=mock_config)
    assert result == mock_db

    # Check first argument (database name)
    assert mock_client.get_database.call_args[0][0] == mock_config.mongo.database

    # Second call returns cached
    result2 = await mongo.get_db(client=mock_client, app_config=mock_config)
    assert result2 == mock_db
    assert mock_client.get_database.call_count == 1


@pytest.mark.asyncio
async def test_check_connection_success(mocker):
    mock_client = mocker.MagicMock()
    mock_config = mocker.Mock()
    mock_client.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    # Mock get_db to prevent side effects and focus on check_connection logic
    mock_get_db = mocker.patch("app.common.mongo.get_db", new_callable=mocker.AsyncMock)

    await mongo.check_connection(mock_client, mock_config)

    mock_get_db.assert_awaited_once_with(mock_client, mock_config)
    mock_client.admin.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_check_connection_failure(mocker):
    mock_client = mocker.MagicMock()
    mock_config = mocker.Mock()
    test_exception = Exception("Connection failed")
    mock_client.admin.command = mocker.AsyncMock(side_effect=test_exception)

    mock_get_db = mocker.patch("app.common.mongo.get_db", new_callable=mocker.AsyncMock)

    with pytest.raises(Exception, match="Connection failed") as exc_info:
        await mongo.check_connection(mock_client, mock_config)

    assert exc_info.value == test_exception
    mock_get_db.assert_awaited_once_with(mock_client, mock_config)
    mock_client.admin.command.assert_awaited_once_with("ping")
