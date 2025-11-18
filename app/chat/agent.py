import abc
import logging

from app import config
from app.bedrock import models as bedrock_models
from app.bedrock import service
from app.chat import models

logger = logging.getLogger(__name__)

app_config = config.get_config()


class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(self, question: str, model_name: str) -> list[models.Message]:
        pass


class BedrockChatAgent(AbstractChatAgent):
    def __init__(self, inference_service: service.BedrockInferenceService):
        self.inference_service = inference_service

    async def execute_flow(self, question: str, model_name: str) -> list[models.Message]:
        system_prompt = "You are a DEFRA agent. All communication should be appropriately professional for a UK government service"

        model_config = self._build_model_config(model_name)

        # Convert question to Anthropic message format
        messages = [
            {"role": "user", "content": question}
        ]

        # Call inference service
        response = self.inference_service.invoke_anthropic(
            model_config=model_config,
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

    def _build_model_config(self, model: str) -> bedrock_models.ModelConfig:
        available_models = app_config.bedrock.available_generation_models

        if model not in available_models:
            msg = f"Requested model '{model}' is not supported."
            raise models.UnsupportedModelError(msg)

        model_info = available_models[model]
        guardrails = model_info.guardrails

        return bedrock_models.ModelConfig(
            id=model_info.id,
            guardrail_id=guardrails.guardrail_id if guardrails else None,
            guardrail_version=guardrails.guardrail_version if guardrails else None,
        )
