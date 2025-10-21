import logging

import bson.binary
import bson.codec_options
import fastapi
import pymongo
import pymongo.asynchronous.database

from app import config
from app.common import tls

logger = logging.getLogger(__name__)

app_config = config.get_config()

client: pymongo.AsyncMongoClient | None = None
db: pymongo.asynchronous.database.AsyncDatabase | None = None


async def get_mongo_client() -> pymongo.AsyncMongoClient:
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
                app_config.mongo.uri,
                tlsCAFile=cert,
                uuidRepresentation="standard"
            )
        else:
            logger.info("Creating MongoDB client")
            client = pymongo.AsyncMongoClient(
                app_config.mongo.uri,
                uuidRepresentation="standard"
            )

        logger.info("Testing MongoDB connection to %s", app_config.mongo.uri)
        await check_connection(client)
    return client


async def get_db(client: pymongo.AsyncMongoClient = fastapi.Depends(get_mongo_client)) -> pymongo.asynchronous.database.AsyncDatabase:
    global db
    if db is None:
        # Configure codec options for proper UUID handling
        codec_options = bson.codec_options.CodecOptions(uuid_representation=bson.binary.UuidRepresentation.STANDARD)
        db = client.get_database(app_config.mongo.database, codec_options=codec_options)

        await _ensure_indexes(db)
    return db


async def check_connection(client: pymongo.AsyncMongoClient):
    database = await get_db(client)
    response = await database.command("ping")
    logger.info("MongoDB PING %s", response)


async def _ensure_indexes(db: pymongo.asynchronous.database.AsyncDatabase):
    """Ensure indexes are created on the necessary collections."""
    logger.info("Ensuring MongoDB indexes are present")

    conversation_history = db.conversation_history

    await conversation_history.create_index("conversationId", unique=True)

    logger.info("MongoDB indexes ensured")
