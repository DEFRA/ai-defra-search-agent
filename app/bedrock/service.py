import logging
from typing import Any

from app import config
from app.bedrock import models
from app.common import knowledge

logger = logging.getLogger(__name__)


class BedrockInferenceService:
    def __init__(
        self,
        api_client,
        runtime_client,
        app_config: config.AppConfig,
        knowledge_retriever: knowledge.KnowledgeRetriever | None,
    ):
        self.api_client = api_client
        self.runtime_client = runtime_client
        self.app_config = app_config
        self.knowledge_retriever = knowledge_retriever

    def invoke_anthropic(
        self,
        model_config: models.ModelConfig,
        system_prompt: str,
        messages: list[dict[str, Any]],
        knowledge_group_id: str | None = None,
    ) -> models.EnhancedModelResponse:
        if not messages:
            msg = "Cannot invoke Anthropic model with no messages"
            raise ValueError(msg)

        sources_found: list[models.RagSource] = []
        model_id = model_config.id

        if self.knowledge_retriever and knowledge_group_id:
            rag_docs = self._retrieve_knowledge(messages, knowledge_group_id)
            if rag_docs:
                system_prompt += self._build_context_string(rag_docs)
                sources_found = self._map_docs_to_sources(rag_docs)

        (guardrail_id, guardrail_version) = (
            model_config.guardrail_id,
            model_config.guardrail_version,
        )

        converse_args: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": self.app_config.bedrock.max_response_tokens,
                "temperature": self.app_config.bedrock.default_model_temprature,
            },
        }

        if (guardrail_id is None) ^ (guardrail_version is None):
            msg = "The guardrail ID and version must be provided together"
            raise ValueError(msg)

        if guardrail_id and guardrail_version is not None:
            converse_args["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
            }

        response = self.runtime_client.converse(**converse_args)

        backing_model = self._get_backing_model(model_id)
        if not backing_model:
            msg = f"Backing model not found for model ID: {model_id}"
            raise ValueError(msg)

        output_message = response["output"]["message"]
        usage_info = response["usage"]

        return models.EnhancedModelResponse(
            model_id=model_id,
            content=output_message["content"],
            usage={
                "input_tokens": usage_info["inputTokens"],
                "output_tokens": usage_info["outputTokens"],
            },
            sources=sources_found,
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

    def _retrieve_knowledge(
        self, messages: list[dict[str, Any]], knowledge_group_id: str
    ) -> list[dict[str, Any]]:
        if not self.knowledge_retriever:
            return []

        # TODO: Make more robust - function to filter only last _user_message?
        query = messages[-1]["content"][0]["text"]
        return self.knowledge_retriever.search(group_id=knowledge_group_id, query=query)

    def _build_context_string(self, docs: list[dict[str, Any]]) -> str:
        context_str = "\n\n".join(
            [
                f'<source name="{d["name"]}" id="{i}">\n{d["content"]}\n</source>'
                for i, d in enumerate(docs)
            ]
        )
        return f"\n\n<context>\n{context_str}\n</context>..."

    def _map_docs_to_sources(
        self, docs: list[dict[str, Any]]
    ) -> list[models.RagSource]:
        return [
            models.RagSource(
                name=d["name"],
                location=d["location"],
                snippet=d["content"][:200] + "...",
                score=d["similarity_score"],
            )
            for d in docs
        ]

    def _get_backing_model(self, model_id: str) -> str | None:
        if not model_id.startswith("arn:aws:bedrock"):
            return model_id

        profile = self.get_inference_profile_details(model_id)

        return profile.models[0]["modelArn"].split("/")[-1]
