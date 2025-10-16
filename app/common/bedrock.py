import json
from dataclasses import dataclass
from logging import getLogger

import boto3

from app.config import get_config

bedrock_client: boto3.client = None

logger = getLogger(__name__)

settings = get_config()


@dataclass
class Message:
    role: str
    content: dict[str, any]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    content: list[dict[str, any]]
    token_usage: TokenUsage


class BedrockInferenceService:
    def __init__(self):
        self.client = get_bedrock_client()

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

        if settings.bedrock.guardrail_identifier and settings.bedrock.guardrail_version:
            logger.info("Using Bedrock guardrail: %s (version: %s)",
                        settings.bedrock.guardrail_identifier,
                        settings.bedrock.guardrail_version)

            invoke_args["guardrailIdentifier"] = settings.bedrock.guardrail_identifier
            invoke_args["guardrailVersion"] = settings.bedrock.guardrail_version
            invoke_args["trace"] = "enabled"

        response = self.client.invoke_model(**invoke_args)

        response_json = json.loads(response["body"].read().decode("utf-8"))

        return ModelResponse(
            content=response_json["content"],
            token_usage=TokenUsage(
                input_tokens=response_json.get("inputTokens", None),
                output_tokens=response_json.get("outputTokens", None),
                total_tokens=response_json.get("totalTokens", None)
            )
        )


def _create_bedrock_client():
    if settings.bedrock.use_credentials:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.bedrock.access_key_id,
            aws_secret_access_key=settings.bedrock.secret_access_key,
            region_name=settings.aws_region
        )

    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region
    )


def get_bedrock_client():
    global bedrock_client

    if bedrock_client is None:
        bedrock_client = _create_bedrock_client()

    return bedrock_client
