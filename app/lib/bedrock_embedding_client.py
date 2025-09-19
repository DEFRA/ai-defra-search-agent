import boto3
from langchain_aws import BedrockEmbeddings

from app.config import config as settings

MODEL = settings.AWS_BEDROCK_EMBEDDING_MODEL
USE_CREDENTIALS = settings.AWS_USE_CREDENTIALS_BEDROCK == "true"
GUARDRAIL = settings.AWS_BEDROCK_GUARDRAIL
GUARDRAIL_VERSION = settings.AWS_BEDROCK_GUARDRAIL_VERSION
PROVIDER = settings.AWS_BEDROCK_PROVIDER

if USE_CREDENTIALS:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_REGION_BEDROCK,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID_BEDROCK,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY_BEDROCK,
    )
else:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
    )


def embedding_bedrock():
    if USE_CREDENTIALS:
        return BedrockEmbeddings(client=bedrock_runtime, model_id=MODEL)

    return BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=MODEL,
        provider=PROVIDER,
        guardrails={
            "guardrailIdentifier": GUARDRAIL,
            "guardrailVersion": GUARDRAIL_VERSION,
            "trace": "enabled",
        },
    )
