import boto3
from botocore.exceptions import ClientError

from app.config import config as settings


class S3Service:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        self.s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            endpoint_url=settings.localstack_url,
        )

    def get_file(self, key: str) -> bytes:
        """Retrieve a file from S3 bucket by key."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except ClientError as e:
            msg = f"Failed to retrieve {key} from S3: {e}"
            raise RuntimeError(msg) from e
