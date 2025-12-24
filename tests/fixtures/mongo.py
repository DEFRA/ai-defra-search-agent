import pytest
from testcontainers.mongodb import MongoDbContainer


class MongoFixture:
    def __init__(self, db, client):
        self.db = db
        self.client = client

    async def close(self):
        """Close the MongoDB client connection."""
        if self.client:
            await self.client.close()


@pytest.fixture(scope="session")
def mongo_container():
    with MongoDbContainer("mongo:6.0.13") as mongo:
        yield mongo


@pytest.fixture(scope="session")
def mongo_uri(mongo_container):
    return mongo_container.get_connection_url()
