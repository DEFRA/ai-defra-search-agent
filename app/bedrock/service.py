import json
import logging

from app import config
from app.bedrock import models

logger = logging.getLogger(__name__)

app_config = config.get_config()


class BedrockInferenceService:
    def __init__(self, api_client, runtime_client):
        self.api_client = api_client
        self.runtime_client = runtime_client

    def invoke_anthropic(
        self, model: str, system_prompt: str, messages: list[dict[str, any]]
    ) -> models.ModelResponse:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": app_config.bedrock.max_response_tokens,
            "temperature": app_config.bedrock.default_model_temprature,
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

        return models.ModelResponse(
            model=self._get_backing_model(model) or model,
            content=response_json["content"],
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

        return models.InferenceProfile(
            id=inference_profile["inferenceProfileId"],
            name=inference_profile["inferenceProfileName"],
            models=inference_profile["models"],
        )

    def _get_backing_model(self, model_id: str) -> str | None:
        if not model_id.startswith("arn:aws:bedrock"):
            return model_id

        return self.get_inference_profile_details(model_id).name
