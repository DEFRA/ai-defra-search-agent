import json
from typing import Any

from langchain_aws import BedrockEmbeddings

from app.common.bedrock import get_bedrock_client


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

            response = self.client.invoke_model(**invoke_args)
            return json.loads(response["body"].read())

        except Exception as e:
            error_msg = f"Failed to invoke Bedrock model {self.model_id}: {str(e)}"
            raise RuntimeError(error_msg) from e


def embedding_bedrock():
    return CustomBedrockEmbeddings(
        client=get_bedrock_client(),
    )
