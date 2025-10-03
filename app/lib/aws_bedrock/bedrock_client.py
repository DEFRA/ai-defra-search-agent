from langchain_aws import ChatBedrockConverse

from app.common.bedrock import get_bedrock_client
from app.config import config as settings


def chat_bedrock_client(model: str = settings.bedrock.generation_model):
    if model is None:
        msg = "Model ID cannot be None. Please check your configuration."
        raise ValueError(msg)

    kwargs = {
        "model_id": model,
        "provider": settings.bedrock.provider
    }

    if settings.bedrock.guardrail_identifier and settings.bedrock.guardrail_version:
        kwargs["guardrail_id"] = settings.bedrock.guardrail_identifier
        kwargs["guardrail_version"] = settings.bedrock.guardrail_version

        kwargs["guardrails"] = {
            "guardrailIdentifier": settings.bedrock.guardrail_identifier,
            "guardrailVersion": settings.bedrock.guardrail_version,
            "trace": "enabled",
        }

    return ChatBedrockConverse(
        client=get_bedrock_client(),
        **kwargs
    )


def chat_bedrock(question, callback):
    llm = chat_bedrock_client()

    if callback:
        return llm.invoke(question, callback=callback)

    return llm.invoke(question)
