import logging
from typing import Any

import bson.binary
import bson.codec_options
import fastapi
import pymongo
import pymongo.asynchronous.database

from app import config, dependencies
from app.common import tls

logger = logging.getLogger(__name__)

client: pymongo.AsyncMongoClient | None = None
db: pymongo.asynchronous.database.AsyncDatabase | None = None


async def get_mongo_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> pymongo.AsyncMongoClient:
    global client
    if client is None:
        # Use the custom CA Certs from env vars if set.
        # We can remove this once we migrate to mongo Atlas.
        cert = tls.custom_ca_certs.get(app_config.mongo.truststore)
        if cert:
            logger.info(
                "Creating MongoDB client with custom TLS cert %s",
                app_config.mongo.truststore,
            )
            client = pymongo.AsyncMongoClient(
                app_config.mongo.uri, tlsCAFile=cert, uuidRepresentation="standard"
            )
        else:
            logger.info("Creating MongoDB client")
            client = pymongo.AsyncMongoClient(
                app_config.mongo.uri, uuidRepresentation="standard"
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
        # TODO: remove "Any" type by defining custom types for the databases collections.
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

    conversation_history = db.conversation_history

    await conversation_history.create_index("conversationId", unique=True)

    logger.info("MongoDB indexes ensured")
