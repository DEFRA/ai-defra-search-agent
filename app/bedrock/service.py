import logging
from typing import Any

import fastapi
from botocore.exceptions import ClientError

from app import config
from app.bedrock import models

logger = logging.getLogger(__name__)


class BedrockInferenceService:
    def __init__(self, api_client, runtime_client, app_config: config.AppConfig):
        self.api_client = api_client
        self.runtime_client = runtime_client
        self.app_config = app_config

    def invoke_anthropic(
        self,
        model_config: models.ModelConfig,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> models.ModelResponse:
        converse_args = self._build_converse_args(model_config, system_prompt, messages)
        response = self._call_bedrock_api(converse_args)
        backing_model = self._get_backing_model(model_config.id)

        if not backing_model:
            msg = f"Backing model not found for model ID: {model_config.id}"
            raise ValueError(msg)

        return self._build_model_response(response, backing_model)

    def _build_converse_args(
        self,
        model_config: models.ModelConfig,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        guardrail_id = model_config.guardrail_id
        guardrail_version = model_config.guardrail_version

        if (guardrail_id is None) ^ (guardrail_version is None):
            msg = "The guardrail ID and version must be provided together"
            raise ValueError(msg)

        converse_args: dict[str, Any] = {
            "modelId": model_config.id,
            "messages": messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": self.app_config.bedrock.max_response_tokens,
                "temperature": self.app_config.bedrock.default_model_temprature,
            },
        }

        if guardrail_id and guardrail_version is not None:
            converse_args["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
            }

        return converse_args

    def _call_bedrock_api(self, converse_args: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.runtime_client.converse(**converse_args)
        except ClientError as e:
            self._handle_bedrock_error(e)
            raise

    def _handle_bedrock_error(self, error: ClientError) -> None:
        error_code = error.response.get("Error", {}).get("Code", "")
        status_code = error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode", 500
        )
        error_message = error.response.get("Error", {}).get(
            "Message", "AWS Bedrock request failed"
        )

        logger.error(
            "Bedrock API error: %s - %s (HTTP %s)",
            error_code,
            error_message,
            status_code,
        )

        # Handle all 4xx client errors (400-499)
        if 400 <= status_code < 500:
            raise fastapi.HTTPException(
                status_code=status_code,
                detail=f"Invalid request to AI model: {error_message}",
            ) from error

        # Re-raise other errors as 500
        raise fastapi.HTTPException(
            status_code=500, detail="AI model request failed"
        ) from error

    def _build_model_response(
        self, response: dict[str, Any], backing_model: str
    ) -> models.ModelResponse:
        output_message = response["output"]["message"]
        usage_info = response["usage"]

        return models.ModelResponse(
            model_id=backing_model,
            content=output_message["content"],
            usage={
                "input_tokens": usage_info["inputTokens"],
                "output_tokens": usage_info["outputTokens"],
            },
        )

    def get_inference_profile_details(
        self, inference_profile_id: str
    ) -> models.InferenceProfile:
        if not inference_profile_id.startswith("arn:aws:bedrock"):
            msg = f"Invalid inference profile ID format: {inference_profile_id}"
            raise ValueError(msg)

        inference_profile = self.api_client.get_inference_profile(
            inferenceProfileIdentifier=inference_profile_id
        )

        if not inference_profile:
            msg = f"Inference profile not found: {inference_profile_id}"
            raise ValueError(msg)

        return models.InferenceProfile(
            id=inference_profile["inferenceProfileId"],
            name=inference_profile["inferenceProfileName"],
            models=inference_profile["models"],
        )

    def _get_backing_model(self, model_id: str) -> str | None:
        if not model_id.startswith("arn:aws:bedrock"):
            return model_id

        profile = self.get_inference_profile_details(model_id)

        return profile.models[0]["modelArn"].split("/")[-1]
