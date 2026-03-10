import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import bson.binary
import bson.codec_options
import fastapi
import pymongo
import pymongo.asynchronous.database
import pymongo.errors

from app import config, dependencies
from app.common import tls

logger = logging.getLogger(__name__)

client: pymongo.AsyncMongoClient | None = None
db: pymongo.asynchronous.database.AsyncDatabase | None = None


class MongoUnavailableError(Exception):
    """Raised when MongoDB is unreachable after all retry attempts."""


async def retry_mongo_operation[T](
    operation: Callable[[], Awaitable[T]],
    retry_attempts: int,
    retry_base_delay_seconds: float,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(retry_attempts):
        try:
            return await operation()
        except pymongo.errors.ConnectionFailure as exc:
            last_exc = exc
            if attempt < retry_attempts - 1:
                delay = retry_base_delay_seconds * (2**attempt)
                logger.warning(
                    "MongoDB transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "MongoDB connection failed after %d attempts: %s",
                    retry_attempts,
                    exc,
                )
        except pymongo.errors.PyMongoError:
            logger.exception("MongoDB error")
            raise
    msg = "MongoDB unavailable"
    raise MongoUnavailableError(msg) from last_exc


def _client_kwargs(app_config: config.AppConfig) -> dict[str, Any]:
    return {
        "uuidRepresentation": "standard",
        "serverSelectionTimeoutMS": app_config.mongo.server_selection_timeout_ms,
        "connectTimeoutMS": app_config.mongo.connect_timeout_ms,
        "socketTimeoutMS": app_config.mongo.socket_timeout_ms,
    }


async def get_mongo_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> pymongo.AsyncMongoClient:
    global client
    if client is None:
        cert = tls.custom_ca_certs.get(app_config.mongo.truststore)
        if cert:
            logger.info(
                "Creating MongoDB client with custom TLS cert %s",
                app_config.mongo.truststore,
            )
            client = pymongo.AsyncMongoClient(
                app_config.mongo.uri, tlsCAFile=cert, **_client_kwargs(app_config)
            )
        else:
            logger.info("Creating MongoDB client")
            client = pymongo.AsyncMongoClient(
                app_config.mongo.uri, **_client_kwargs(app_config)
            )

        logger.info("Testing MongoDB connection to %s", app_config.mongo.uri)
        await check_connection(client, app_config)
    return client


async def get_db(
    client: pymongo.AsyncMongoClient = fastapi.Depends(get_mongo_client),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> pymongo.asynchronous.database.AsyncDatabase:
    global db
    if db is None:
        codec_options: bson.codec_options.CodecOptions[dict[str, Any]] = (
            bson.codec_options.CodecOptions(
                uuid_representation=bson.binary.UuidRepresentation.STANDARD
            )
        )

        db = client.get_database(app_config.mongo.database, codec_options=codec_options)

        await _ensure_indexes(db)
    return db


async def check_connection(
    client: pymongo.AsyncMongoClient, app_config: config.AppConfig
):
    await get_db(client, app_config)
    try:
        await client.admin.command("ping")
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.error("MongoDB connection failed: %s", e)
        raise e


async def _ensure_indexes(db: pymongo.asynchronous.database.AsyncDatabase):
    """Ensure indexes are created on the necessary collections."""
    logger.info("Ensuring MongoDB indexes are present")

    conversations = db.conversations

    await conversations.create_index("conversation_id", unique=True)

    logger.info("MongoDB indexes ensured")
