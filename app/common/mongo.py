from logging import getLogger

from fastapi import Depends
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from bson.codec_options import CodecOptions
from bson.binary import UuidRepresentation

from app.common.tls import custom_ca_certs
from app.config import config

logger = getLogger(__name__)

client: AsyncMongoClient | None = None
db: AsyncDatabase | None = None


async def get_mongo_client() -> AsyncMongoClient:
    global client
    if client is None:
        # Use the custom CA Certs from env vars if set.
        # We can remove this once we migrate to mongo Atlas.
        cert = custom_ca_certs.get(config.mongo.truststore)
        if cert:
            logger.info(
                "Creating MongoDB client with custom TLS cert %s",
                config.mongo.truststore,
            )
            client = AsyncMongoClient(
                config.mongo.uri, 
                tlsCAFile=cert,
                uuidRepresentation="standard"
            )
        else:
            logger.info("Creating MongoDB client")
            client = AsyncMongoClient(
                config.mongo.uri,
                uuidRepresentation="standard"
            )

        logger.info("Testing MongoDB connection to %s", config.mongo.uri)
        await check_connection(client)
    return client


async def get_db(client: AsyncMongoClient = Depends(get_mongo_client)) -> AsyncDatabase:
    global db
    if db is None:
        # Configure codec options for proper UUID handling
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        db = client.get_database(config.mongo.database, codec_options=codec_options)

        await _ensure_indexes(db)
    return db


async def check_connection(client: AsyncMongoClient):
    database = await get_db(client)
    response = await database.command("ping")
    logger.info("MongoDB PING %s", response)


async def _ensure_indexes(db: AsyncDatabase):
    """Ensure indexes are created on the necessary collections."""
    logger.info("Ensuring MongoDB indexes are present")

    conversation_history = db.conversation_history

    await conversation_history.create_index("conversationId", unique=True)

    logger.info("MongoDB indexes ensured")