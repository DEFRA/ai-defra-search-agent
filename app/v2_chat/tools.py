import abc

from app.common import http_client


class AbstractKnowledgeSearchTool(abc.ABC):
    @abc.abstractmethod
    async def search(self, query: str) -> list[dict[str, any]]:
        pass


class KnowledgeSearchTool(AbstractKnowledgeSearchTool):
    def __init__(self):
        self._client = http_client.create_async_client()

