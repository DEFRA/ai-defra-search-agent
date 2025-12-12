import abc
import logging

from app import config
from app.bedrock import models as bedrock_models
from app.bedrock import service
from app.chat import models

logger = logging.getLogger(__name__)


class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(self, question: str, model_id: str) -> list[models.Message]:
        pass


class BedrockChatAgent(AbstractChatAgent):
    def __init__(
        self,
        inference_service: service.BedrockInferenceService,
        app_config: config.AppConfig,
    ):
        self.inference_service = inference_service
        self.app_config = app_config

    async def execute_flow(self, question: str, model_id: str) -> list[models.Message]:
        system_prompt = "You are a DEFRA agent. All communication should be appropriately professional for a UK government service"

        model_config = self._build_model_config(model_id)

        messages = [
            models.UserMessage(
                content=question,
                model_id=model_config.id,
                model_name=self.app_config.bedrock.available_generation_models[
                    model_id
                ].name,
            ).to_dict()
        ]

        response = self.inference_service.invoke_anthropic(
            model_config=model_config,
            system_prompt=system_prompt,
            messages=messages,
        )

        input_tokens = response.usage["input_tokens"]
        output_tokens = response.usage["output_tokens"]
        usage = models.TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return [
            models.AssistantMessage(
                content=content_block["text"],
                model_id=model_id,
                model_name=self.app_config.bedrock.available_generation_models[
                    model_id
                ].name,
                usage=usage,
            )
            for content_block in response.content
        ]

    def _build_model_config(self, model: str) -> bedrock_models.ModelConfig:
        available_models = self.app_config.bedrock.available_generation_models

        if model not in available_models:
            msg = f"Requested model '{model}' is not supported."
            raise models.UnsupportedModelError(msg)

        model_info = available_models[model]
        guardrails = model_info.guardrails

        return bedrock_models.ModelConfig(
            id=model_info.bedrock_model_id,
            guardrail_id=guardrails.guardrail_id if guardrails else None,
            guardrail_version=guardrails.guardrail_version if guardrails else None,
        )
