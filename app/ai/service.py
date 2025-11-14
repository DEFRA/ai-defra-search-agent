import dataclasses
import json
import logging

from app import config

logger = logging.getLogger(__name__)

app_config = config.get_config()


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    model: str
    content: list[dict[str, any]]


class BedrockInferenceService:
    def __init__(self, api_client, runtime_client):
        self.api_client = api_client
        self.runtime_client = runtime_client

    def invoke_anthropic(
        self, model: str, system_prompt: str, messages: list[dict[str, any]]
    ) -> ModelResponse:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": app_config.bedrock.max_response_tokens,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": messages,
        }

        invoke_args = {"modelId": model, "body": json.dumps(native_request)}

        if (
            app_config.bedrock.guardrail_identifier
            and app_config.bedrock.guardrail_version
        ):
            invoke_args["guardrailIdentifier"] = app_config.bedrock.guardrail_identifier
            invoke_args["guardrailVersion"] = app_config.bedrock.guardrail_version

        response = self.runtime_client.invoke_model(**invoke_args)

        response_json = json.loads(response["body"].read().decode("utf-8"))

        return ModelResponse(
            model=self._get_backing_model(model) or model,
            content=response_json["content"],
            token_usage=self._extract_token_usage(response_json),
        )