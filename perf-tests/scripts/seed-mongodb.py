#!/usr/bin/env python3
"""Seed MongoDB with perf-test data."""

import json
import logging
import pathlib

import pymongo.asynchronous.database

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


async def seed(db: pymongo.asynchronous.database.AsyncDatabase) -> None:
    """Seed MongoDB with perf-test data if data files are present."""
    if not DATA_DIR.exists():
        logger.info("Perf-test data directory not found, skipping seed")
        return

    for collection_name, filename in [
        ("knowledgeGroups", "knowledgeGroups.json"),
        ("knowledgeSnapshots", "knowledgeSnapshots.json"),
    ]:
        data_file = DATA_DIR / filename
        if not data_file.exists():
            logger.warning("Data file not found: %s, skipping seed", data_file)
            return

        collection = db.get_collection(collection_name)
        await collection.drop()
        await collection.insert_one(json.loads(data_file.read_text()))
        logger.info("Seeded collection: %s", collection_name)
