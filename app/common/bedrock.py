import json
import boto3

from dataclasses import dataclass, field
from logging import getLogger

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
    model: str
    content: list[dict[str, any]] = field(default_factory=list)
    token_usage: TokenUsage = TokenUsage(0, 0, 0)


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

        if settings.bedrock.guardrail_identifier and settings.bedrock.guardrail_version:
            invoke_args["guardrailIdentifier"] = settings.bedrock.guardrail_identifier
            invoke_args["guardrailVersion"] = settings.bedrock.guardrail_version

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

        profileModels = inference_profile.get("models", [])

        if len(profileModels) == 0:
            return None

        model_arn = profileModels[0].get("modelArn", None)

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


def _create_bedrock_client():
    if settings.bedrock.use_credentials:
        return boto3.client(
            "bedrock",
            aws_access_key_id=settings.bedrock.access_key_id,
            aws_secret_access_key=settings.bedrock.secret_access_key,
            region_name=settings.aws_region
        )

    return boto3.client(
        "bedrock",
        region_name=settings.aws_region
    )


def get_bedrock_runtime_client():
    global bedrock_client

    if bedrock_client is None:
        bedrock_client = _create_bedrock_runtime_client()

    return bedrock_client

def get_bedrock_client():
    global bedrock_client

    if bedrock_client is None:
        bedrock_client = _create_bedrock_client()

    return bedrock_client
