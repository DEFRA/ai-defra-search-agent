import dataclasses
import json
import logging

import boto3

from app import config

bedrock_runtime_client: boto3.client = None
bedrock_client: boto3.client = None

logger = logging.getLogger(__name__)

app_config = config.get_config()


@dataclasses.dataclass
class Message:
    role: str
    content: dict[str, any]


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass(frozen=True)
class ChatResponse:
    content: list[dict[str, any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    model: str
    content: list[dict[str, any]]
    token_usage: TokenUsage


class BedrockInferenceService:
    def __init__(self):
        self.runtime_client = get_bedrock_runtime_client()
        self.api_client = get_bedrock_client()

    def invoke_anthropic(self, model: str, system_prompt: str, messages: list[Message]) -> ModelResponse:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": messages
        }

        invoke_args = {
            "modelId": model,
            "body": json.dumps(native_request)
        }

        if app_config.bedrock.guardrail_identifier and app_config.bedrock.guardrail_version:
            invoke_args["guardrailIdentifier"] = app_config.bedrock.guardrail_identifier
            invoke_args["guardrailVersion"] = app_config.bedrock.guardrail_version

        response = self.runtime_client.invoke_model(**invoke_args)

        response_json = json.loads(response["body"].read().decode("utf-8"))

        return ModelResponse(
            model=self._get_backing_model(model) or model,
            content=response_json["content"],
            token_usage=self._extract_token_usage(response_json)
        )


    def _get_backing_model(self, model_id: str) -> str | None:
        if not model_id.startswith("arn:aws:bedrock:"):
            return model_id

        inference_profile = self.api_client.get_inference_profile(
            inferenceProfileIdentifier=model_id
        )

        profile_models = inference_profile.get("models", [])

        if len(profile_models) == 0:
            return None

        model_arn = profile_models[0].get("modelArn", None)

        return model_arn.split("/")[-1]

    def _extract_token_usage(self, response: dict) -> TokenUsage:
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

def _create_bedrock_runtime_client():
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region
        )

    return boto3.client(
        "bedrock-runtime",
        region_name=app_config.aws_region
    )


def _create_bedrock_client():
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region
        )

    return boto3.client(
        "bedrock",
        region_name=app_config.aws_region
    )


def get_bedrock_runtime_client():
    global bedrock_runtime_client

    if bedrock_runtime_client is None:
        bedrock_runtime_client = _create_bedrock_runtime_client()

    return bedrock_runtime_client

def get_bedrock_client():
    global bedrock_client

    if bedrock_client is None:
        bedrock_client = _create_bedrock_client()

    return bedrock_client
