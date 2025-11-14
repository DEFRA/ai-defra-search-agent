import abc

import pymongo.asynchronous.database

from app.chat import models


class AbstractConversationRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, conversation: models.Conversation) -> None:
        pass

    @abc.abstractmethod
    async def get(self, conversation_id: str) -> models.Conversation:
        pass


class MongoConversationRepository(AbstractConversationRepository):
    def __init__(self, db: pymongo.asynchronous.database.AsyncDatabase):
        pass

    async def save(self, conversation: models.Conversation) -> None:
        pass

    async def get(self, conversation_id: str) -> models.Conversation:
        pass
