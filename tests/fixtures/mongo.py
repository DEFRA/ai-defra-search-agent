import pymongo
import pytest


@pytest.fixture
def db():
    client = pymongo.AsyncMongoClient("mongo", uuidRepresentation="standard")

    return client.get_database("ai_defra_search_agent")
