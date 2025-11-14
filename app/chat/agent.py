import abc

from app import config
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
        self, question: str
    ) -> list[models.Message]:
        model = config.get_config().bedrock.default_generation_model
        system_prompt = "You are a DEFRA agent. All communication should be appropriately professional for a UK government service"

        # Convert question to Anthropic message format
        messages = [
            {"role": "user", "content": question}
        ]

        # Call inference service
        response = self.inference_service.invoke_anthropic(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
        )

        # Convert response to list of messages
        result_messages = []
        for content_block in response.content:
            message = models.Message(
                role="assistant",
                content=content_block["text"],
                model=response.model,
            )
            result_messages.append(message)

        return result_messages
