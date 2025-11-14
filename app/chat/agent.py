import abc

from app.bedrock import service
from app.chat import models


class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(
        self, question: str
    ) -> list[models.Message]:
        pass


class BedrockChatAgent(AbstractChatAgent):
    def __init__(self, inference_service: service.BedrockInferenceService):
        self.inference_service = inference_service

    async def execute_flow(
        self, conversation: models.Conversation # noqa: ARG002
    ) -> list[models.Message]:
        return [
            models.Message(
                role="assistant",
                content="mock response",
                model=None,
            )
        ]
