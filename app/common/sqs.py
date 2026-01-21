import json

from aiobotocore.session import get_session

from app import config


class SQSClient:
    """Async SQS client using aiobotocore for non-blocking operations."""

    def __init__(self):
        self.session = get_session()
        self.queue_url = config.config.chat_queue_url
        self.region_name = config.config.aws_region
        self.endpoint_url = config.config.localstack_url
        self._client = None

    async def __aenter__(self):
        """Create async SQS client on context manager entry."""
        self._client = await self.session.create_client(
            "sqs",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        ).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async SQS client on context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def send_message(self, message_body: dict) -> str:
        """Send a message to the SQS queue."""
        response = await self._client.send_message(
            QueueUrl=self.queue_url, MessageBody=json.dumps(message_body)
        )
        return response["MessageId"]

    async def receive_messages(
        self, max_messages: int = 1, wait_time: int = 20
    ) -> list:
        """Receive messages from the SQS queue with long polling."""
        response = await self._client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
        )
        return response.get("Messages", [])

    async def delete_message(self, receipt_handle: str) -> None:
        """Delete a message from the SQS queue."""
        await self._client.delete_message(
            QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
        )
