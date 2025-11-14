import pymongo
import pytest


class MongoFixture:
    def __init__(self, db, client):
        self.db = db
        self.client = client
        
    async def close(self):
        """Close the MongoDB client connection."""
        if self.client:
            await self.client.close()


@pytest.fixture
async def mongo():
    client = pymongo.AsyncMongoClient("mongodb", uuidRepresentation="standard")
    db = client.get_database("ai_defra_search_agent")
    fixture = MongoFixture(db, client)
    
    yield fixture
    
    # Clean up: close the client connection
    await fixture.close()
