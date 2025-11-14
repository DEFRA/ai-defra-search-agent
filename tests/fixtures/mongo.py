import pymongo
import pytest


class MongoFixture:
    def __init__(self, db):
        self.db = db


@pytest.fixture
def mongo():
    client = pymongo.AsyncMongoClient("mongo", uuidRepresentation="standard")
    db = client.get_database("ai_defra_search_agent")
    return MongoFixture(db)
