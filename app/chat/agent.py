import abc

from app.chat import models

class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(self, question: str, conversation: models.Conversation) -> models.ChatState:
        pass
