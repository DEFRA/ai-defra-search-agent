import json
from typing import Any

import boto3
from langchain_aws import BedrockEmbeddings

from app.config import config as settings

MODEL = settings.AWS_BEDROCK_EMBEDDING_MODEL or "amazon.titan-embed-text-v2:0"
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


class CustomBedrockEmbeddings(BedrockEmbeddings):
    def _invoke_model(self, input_body: dict[str, Any] = None) -> dict[str, Any]:
        if input_body is None:
            input_body = {}

        if self.model_kwargs:
            input_body = {**input_body, **self.model_kwargs}

        body = json.dumps(input_body)

        try:
            invoke_args = {
                "body": body,
                "modelId": self.model_id,
                "accept": "application/json",
                "contentType": "application/json",
            }
            if GUARDRAIL is not None:
                invoke_args["guardrailIdentifier"] = GUARDRAIL
            if GUARDRAIL_VERSION is not None:
                invoke_args["guardrailVersion"] = GUARDRAIL_VERSION

            response = bedrock_runtime.invoke_model(**invoke_args)
            return json.loads(response["body"].read())

        except Exception as e:
            error_msg = f"Failed to invoke Bedrock model {self.model_id}: {str(e)}"
            raise RuntimeError(error_msg) from e


def embedding_bedrock():
    return CustomBedrockEmbeddings(
        client=bedrock_runtime, model_id=MODEL, normalize=True, provider=PROVIDER
    )
