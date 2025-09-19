from langchain_aws import ChatBedrockConverse

from app.config import config as settings

USE_CREDENTIALS = settings.AWS_USE_CREDENTIALS_BEDROCK == "true"
MODEL_ID = settings.AWS_BEDROCK_MODEL
GUARDRAIL = settings.AWS_BEDROCK_GUARDRAIL
GUARDRAIL_VERSION = settings.AWS_BEDROCK_GUARDRAIL_VERSION
PROVIDER = settings.AWS_BEDROCK_PROVIDER


def chat_bedrock_client(model: str = MODEL_ID):
    if model is None:
        error_message = "Model ID cannot be None. Please check your configuration."
        raise ValueError(error_message)

    if USE_CREDENTIALS:
        print("USE CREDENTIALS")
        llm = ChatBedrockConverse(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID_BEDROCK,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY_BEDROCK,
            region_name=settings.AWS_REGION_BEDROCK,
            model_id=model,
        )

    else:
        llm = ChatBedrockConverse(model_id=model)

    if GUARDRAIL and GUARDRAIL_VERSION:
        llm.guardrail_config = {
            "guardrailIdentifier": GUARDRAIL,
            "guardrailVersion": GUARDRAIL_VERSION,
            "trace": "enabled",
        }

    llm.provider = "anthropic"

    return llm


def chat_bedrock(question, callback):
    llm = chat_bedrock_client()

    if callback:
        return llm.invoke(question, callback=callback)

    return llm.invoke(question)
