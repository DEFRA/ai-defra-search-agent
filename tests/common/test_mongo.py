import pymongo.errors
import pytest

from app.common import mongo

_SERVER_SELECTION_TIMEOUT_MS = 5000
_CONNECT_TIMEOUT_MS = 5000
_SOCKET_TIMEOUT_MS = 10000
_RETRY_ATTEMPTS = 2
_RETRY_BASE_DELAY_SECONDS = 0.5


# Reset the global client variable before each test
@pytest.fixture(autouse=True)
def reset_mongo_client():
    mongo.client = None
    mongo.db = None
    yield
    mongo.client = None
    mongo.db = None


def _make_mock_config(mocker, *, uri="mongodb://localhost:27017", truststore=None):
    mock_config = mocker.Mock()
    mock_config.mongo.uri = uri
    mock_config.mongo.truststore = truststore
    mock_config.mongo.server_selection_timeout_ms = _SERVER_SELECTION_TIMEOUT_MS
    mock_config.mongo.connect_timeout_ms = _CONNECT_TIMEOUT_MS
    mock_config.mongo.socket_timeout_ms = _SOCKET_TIMEOUT_MS
    mock_config.mongo.retry_attempts = _RETRY_ATTEMPTS
    mock_config.mongo.retry_base_delay_seconds = _RETRY_BASE_DELAY_SECONDS
    mock_config.mongo.database = "test_db"
    return mock_config


@pytest.mark.asyncio
async def test_get_mongo_client_initialization(mocker):
    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")
    mock_instance = mock_client_cls.return_value

    mock_config = _make_mock_config(mocker)

    mock_db = mocker.MagicMock()
    mock_db.conversations.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})
    mock_instance.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    client = await mongo.get_mongo_client(app_config=mock_config)

    assert client == mock_instance
    mock_client_cls.assert_called_once_with(
        mock_config.mongo.uri,
        uuidRepresentation="standard",
        serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=_CONNECT_TIMEOUT_MS,
        socketTimeoutMS=_SOCKET_TIMEOUT_MS,
    )
    mock_instance.admin.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_get_mongo_client_with_custom_tls(mocker):
    mock_config = _make_mock_config(mocker, truststore="custom-cert-key")

    mocker.patch.dict(
        "app.common.tls.custom_ca_certs", {"custom-cert-key": "/path/to/cert.pem"}
    )

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")
    mock_instance = mock_client_cls.return_value
    mock_db = mocker.MagicMock()
    mock_db.conversations.create_index = mocker.AsyncMock()
    mock_instance.get_database.return_value = mock_db
    mock_db.command = mocker.AsyncMock(return_value={"ok": 1})
    mock_instance.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    await mongo.get_mongo_client(app_config=mock_config)

    mock_client_cls.assert_called_once_with(
        mock_config.mongo.uri,
        tlsCAFile="/path/to/cert.pem",
        uuidRepresentation="standard",
        serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=_CONNECT_TIMEOUT_MS,
        socketTimeoutMS=_SOCKET_TIMEOUT_MS,
    )


@pytest.mark.asyncio
async def test_get_mongo_client_returns_existing(mocker):
    existing_client = mocker.Mock()
    mongo.client = existing_client
    mock_config = _make_mock_config(mocker)

    mock_client_cls = mocker.patch("app.common.mongo.pymongo.AsyncMongoClient")

    result = await mongo.get_mongo_client(app_config=mock_config)

    assert result == existing_client
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_get_db(mocker):
    mock_client = mocker.MagicMock()
    mock_db = mocker.Mock()
    mock_config = _make_mock_config(mocker)

    mock_db.conversations.create_index = mocker.AsyncMock()
    mock_client.get_database.return_value = mock_db

    result = await mongo.get_db(client=mock_client, app_config=mock_config)
    assert result == mock_db

    assert mock_client.get_database.call_args[0][0] == mock_config.mongo.database

    result2 = await mongo.get_db(client=mock_client, app_config=mock_config)
    assert result2 == mock_db
    assert mock_client.get_database.call_count == 1


@pytest.mark.asyncio
async def test_check_connection_success(mocker):
    mock_client = mocker.MagicMock()
    mock_config = _make_mock_config(mocker)
    mock_client.admin.command = mocker.AsyncMock(return_value={"ok": 1})

    mock_get_db = mocker.patch("app.common.mongo.get_db", new_callable=mocker.AsyncMock)

    await mongo.check_connection(mock_client, mock_config)

    mock_get_db.assert_awaited_once_with(mock_client, mock_config)
    mock_client.admin.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_check_connection_failure(mocker):
    mock_client = mocker.MagicMock()
    mock_config = _make_mock_config(mocker)
    test_exception = Exception("Connection failed")
    mock_client.admin.command = mocker.AsyncMock(side_effect=test_exception)

    mock_get_db = mocker.patch("app.common.mongo.get_db", new_callable=mocker.AsyncMock)

    with pytest.raises(Exception, match="Connection failed") as exc_info:
        await mongo.check_connection(mock_client, mock_config)

    assert exc_info.value == test_exception
    mock_get_db.assert_awaited_once_with(mock_client, mock_config)
    mock_client.admin.command.assert_awaited_once_with("ping")


@pytest.mark.asyncio
async def test_retry_mongo_operation_retries_on_pymongo_error(mocker):
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    msg = "transient"
    op = mocker.AsyncMock(side_effect=[pymongo.errors.ConnectionFailure(msg), "ok"])

    result = await mongo.retry_mongo_operation(
        op, retry_attempts=3, retry_base_delay_seconds=0.1
    )
    assert result == "ok"
    assert op.call_count == 2


@pytest.mark.asyncio
async def test_retry_mongo_operation_raises_mongo_unavailable_after_exhaustion(mocker):
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    msg = "timeout"
    op = mocker.AsyncMock(
        side_effect=[
            pymongo.errors.ServerSelectionTimeoutError(msg),
            pymongo.errors.ServerSelectionTimeoutError(msg),
        ]
    )

    with pytest.raises(mongo.MongoUnavailableError):
        await mongo.retry_mongo_operation(
            op, retry_attempts=2, retry_base_delay_seconds=0.1
        )


@pytest.mark.asyncio
async def test_retry_mongo_operation_does_not_catch_non_pymongo_errors(mocker):
    msg = "not a mongo error"
    op = mocker.AsyncMock(side_effect=ValueError(msg))

    with pytest.raises(ValueError, match="not a mongo error"):
        await mongo.retry_mongo_operation(
            op, retry_attempts=3, retry_base_delay_seconds=0
        )


@pytest.mark.asyncio
async def test_retry_mongo_operation_sleeps_with_exponential_backoff(mocker):
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    msg = "fail"
    op = mocker.AsyncMock(
        side_effect=[
            pymongo.errors.ConnectionFailure(msg),
            pymongo.errors.ConnectionFailure(msg),
            "done",
        ]
    )

    await mongo.retry_mongo_operation(
        op, retry_attempts=3, retry_base_delay_seconds=1.0
    )

    # First retry: 1.0 * 2^0 = 1.0, second retry: 1.0 * 2^1 = 2.0
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
