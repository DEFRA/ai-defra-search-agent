from datetime import UTC, datetime
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase


class ConversationHistoryService:
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.collection = db.conversation_history

    async def add_message(self, conversation_id: str, message: dict[str, Any]):
        await self.collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"messages": message},
                "$setOnInsert": {"created_at": datetime.now(UTC)},
            },
            upsert=True,
        )

    async def get_history(self, conversation_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"conversation_id": conversation_id})

    async def create_conversation(self, conversation_id: str):
        await self.collection.insert_one(
            {
                "conversation_id": conversation_id,
                "messages": [],
                "created_at": datetime.now(UTC),
            }
        )
