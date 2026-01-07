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
    ) -> models.ModelResponse:
        return models.ModelResponse(
            model_id=model_config.id,
            content=[{"text": "This is a stub response."}],
            usage={"input_tokens": 10, "output_tokens": 15},
        )

    def get_inference_profile_details(
        self, inference_profile_id: str
    ) -> models.InferenceProfile:
        return models.InferenceProfile(
            id=inference_profile_id, name="geni-ai-3.5", models=["geni-ai-3.5"]
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
    def converse(self, **_) -> dict:
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
