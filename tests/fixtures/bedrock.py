from app.bedrock import models, service


class StubBedrockInferenceService(service.BedrockInferenceService):
    def __init__(self):
        self.api_client = None
        self.runtime_client = None

    def invoke_anthropic(
        self, model_config: models.ModelConfig, system_prompt: str, messages: list[dict[str, any]]
    ) -> models.ModelResponse:
        return models.ModelResponse(
            model=model_config.id,
            content=[{"text": "This is a stub response."}]
        )

    def get_inference_profile_details(
        self, inference_profile_id: str
    ) -> models.InferenceProfile:
        return models.InferenceProfile(
            id=inference_profile_id,
            name="geni-ai-3.5",
            models=["geni-ai-3.5"]
        )
