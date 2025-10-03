import boto3

from app.config import config as settings

bedrock_client: boto3.client = None


def _create_bedrock_client():
    if settings.bedrock.use_credentials:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.bedrock.access_key_id,
            aws_secret_access_key=settings.bedrock.secret_access_key,
            region_name=settings.aws_region
        )

    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region
    )


def get_bedrock_client():
    global bedrock_client

    if bedrock_client is None:
        bedrock_client = _create_bedrock_client()

    return bedrock_client

