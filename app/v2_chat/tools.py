from abc import ABC, abstractmethod

from app.common.http_client import create_async_client


class AbstractKnowledgeSearchTool(ABC):
    @abstractmethod
    async def search(self, query: str) -> list[dict[str, any]]:
        pass


class KnowledgeSearchTool(AbstractKnowledgeSearchTool):
    def __init__(self):
        self._client = create_async_client()

        