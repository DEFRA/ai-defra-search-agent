#!/usr/bin/env python3
"""Seed MongoDB with perf-test data. Run before starting the application."""

import json
import logging
import os
import pathlib
import sys

import pymongo
import pymongo.errors

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("MONGO_DATABASE", "ai-defra-search-agent")
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def get_client() -> pymongo.MongoClient:
    truststore_env = os.environ.get("MONGO_TRUSTSTORE", "TRUSTSTORE_CDP_ROOT_CA")
    tls_ca_file = os.environ.get(truststore_env)
    if tls_ca_file:
        logger.info("Connecting with TLS CA file: %s", tls_ca_file)
        return pymongo.MongoClient(MONGO_URI, tlsCAFile=tls_ca_file)
    return pymongo.MongoClient(MONGO_URI)


def seed(client: pymongo.MongoClient) -> None:
    db = client[DB_NAME]

    for collection_name, filename in [
        ("knowledgeGroups", "knowledgeGroups.json"),
        ("knowledgeSnapshots", "knowledgeSnapshots.json"),
    ]:
        data_file = DATA_DIR / filename
        if not data_file.exists():
            logger.error("Required data file not found: %s", data_file)
            sys.exit(1)

        collection = db[collection_name]
        collection.drop()
        logger.info("Dropped collection: %s", collection_name)

        data = json.loads(data_file.read_text())
        collection.insert_one(data)
        logger.info("Seeded collection: %s", collection_name)


def main() -> None:
    logger.info("Connecting to MongoDB: %s", MONGO_URI)
    client = get_client()
    try:
        client.admin.command("ping")
        logger.info("MongoDB connection successful")
    except pymongo.errors.ConnectionFailure as e:
        logger.error("MongoDB connection failed: %s", e)
        sys.exit(1)

    seed(client)
    logger.info("Seeding complete")


if __name__ == "__main__":
    main()
