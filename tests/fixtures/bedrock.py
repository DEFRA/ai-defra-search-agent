import json
from typing import Any

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


class StubBedrockRuntimeClient:
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
