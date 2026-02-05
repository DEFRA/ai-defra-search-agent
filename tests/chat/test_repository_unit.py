import uuid

import pytest

from app.chat import models, repository


class DummyCollection:
    def __init__(self):
        self.storage = {}

    async def update_one(self, query, update, upsert=False):
        # store the update for inspection
        self.last_query = query
        self.last_update = update
        self.last_upsert = upsert

    async def find_one(self, query):
        # return a sample document matching query
        if query.get("conversation_id"):
            return {
                "conversation_id": query["conversation_id"],
                "messages": [
                    {
                        "message_id": uuid.uuid4(),
                        "role": "user",
                        "content": "hi",
                        "model": "m",
                        "model_name": "mname",
                        "status": models.MessageStatus.COMPLETED.value,
                        "timestamp": None,
                        "usage": None,
                    }
                ],
            }
        return None


@pytest.mark.asyncio
async def test_mongo_conversation_repository_save_get_update():
    dummy_db = type("M", (), {})()
    coll = DummyCollection()
    dummy_db.conversations = coll

    repo = repository.MongoConversationRepository(db=dummy_db)

    # Test save (should call update_one)
    conv = models.Conversation()
    conv.add_message(
        models.UserMessage(content="hello", model_id="mid", model_name="mname")
    )
    await repo.save(conv)
    assert coll.last_upsert is True

    # Test get returns Conversation
    cid = uuid.uuid4()
    found = await repo.get(cid)
    assert isinstance(found, models.Conversation)

    # Test update_message_status calls update_one
    await repo.update_message_status(
        cid, uuid.uuid4(), models.MessageStatus.COMPLETED, "err"
    )
    assert coll.last_update is not None
