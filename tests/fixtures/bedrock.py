import json
from typing import Any

import pytest

from app.bedrock import models, service


class StubBedrockInferenceService(service.BedrockInferenceService):
    def __init__(self):
        self.api_client = None
        self.runtime_client = None

    def invoke_anthropic(
        self,
        model_config: models.ModelConfig,
        system_prompt: str,  # noqa: ARG002
        messages: list[dict[str, Any]],  # noqa: ARG002
        knowledge_group_id: str | None = None,  # noqa: ARG002
    ) -> models.EnhancedModelResponse:
        return models.EnhancedModelResponse(
            model_id=model_config.id,
            content=[{"text": "This is a stub response."}],
            usage={"input_tokens": 10, "output_tokens": 15},
            sources=[],
        )

    def get_inference_profile_details(
        self, inference_profile_id: str
    ) -> models.InferenceProfile:
        return models.InferenceProfile(
            id=inference_profile_id, name="geni-ai-3.5", models=[{"id": "geni-ai-3.5"}]
        )


class FakeStreamingBody:
    def __init__(self, content: bytes):
        self._content = content

    def read(self) -> bytes:
        return self._content


class StubBedrockRuntimeBedrockV1Client:
    def invoke_model(self, **kwargs) -> dict:
        response = {
            "id": "stub-response-id",
            "model": kwargs.get("modelId", "unknown-model"),
            "type": "message",
            "role": "assistant",
            "content": [{"text": "This is a stub response."}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 15,
            },
        }

        encoded_response = FakeStreamingBody(
            content=json.dumps(response).encode("utf-8")
        )

        return {"body": encoded_response, "contentType": "application/json"}


class StubBedrockRuntimeBedrockV2Client:
    def __init__(self, raise_exception: str | None = None):
        self.raise_exception = raise_exception

    def converse(self, **_) -> dict:
        if self.raise_exception:
            self._raise_client_error(self.raise_exception)

        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "This is a stub response."}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 15,
                "totalTokens": 25,
            },
            "metrics": {
                "latencyMs": 100,
            },
        }

    def _raise_client_error(self, error_code: str):
        from botocore.exceptions import ClientError

        error_messages = {
            "ThrottlingException": "Rate exceeded",
            "ValidationException": "Invalid input parameters",
            "ModelTimeoutException": "Model request timed out",
            "ModelErrorException": "Model error occurred",
            "AccessDeniedException": "Access denied to the requested resource",
            "ResourceNotFoundException": "The requested resource was not found",
            "ServiceUnavailableException": "Service temporarily unavailable",
            "InternalServerException": "Internal server error occurred",
            "ModelNotReadyException": "Model is not ready",
        }

        error_response = {
            "Error": {
                "Code": error_code,
                "Message": error_messages.get(error_code, "Unknown error occurred"),
            },
            "ResponseMetadata": {
                "RequestId": "stub-request-id",
                "HTTPStatusCode": 400,
            },
        }
        raise ClientError(error_response, "converse")


class StubBedrockClient:
    def get_inference_profile(self, **kwargs) -> dict:
        inference_profile_id = kwargs.get("inferenceProfileIdentifier")

        return {
            "inferenceProfileName": "Stub Inference Profile",
            "description": "This is a stub inference profile.",
            "models": [
                {
                    "modelArn": "arn:aws:bedrock:eu-central-1::foundation-model/geni-ai-3.5",
                }
            ],
            "inferenceProfileArn": f"arn:aws:bedrock:eu-central-1::inference-profile/{inference_profile_id}",
            "inferenceProfileId": inference_profile_id,
            "status": "ACTIVE",
            "type": "APPLICATION",
        }


@pytest.fixture
def bedrock_client():
    return StubBedrockClient()


@pytest.fixture
def bedrock_inference_service():
    return StubBedrockInferenceService()


@pytest.fixture
def bedrock_runtime_v1_client():
    return StubBedrockRuntimeBedrockV1Client()


@pytest.fixture
def bedrock_runtime_v2_client():
    return StubBedrockRuntimeBedrockV2Client()
