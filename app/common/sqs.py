import json
import logging

from aiobotocore.session import get_session

from app import config

logger = logging.getLogger(__name__)

# SQS default polling configuration
DEFAULT_MAX_MESSAGES = 1
DEFAULT_LONG_POLL_WAIT_SECONDS = 20


class SQSClient:
    """Async SQS client using aiobotocore for non-blocking operations."""

    def __init__(self):
        self.session = get_session()
        self.queue_url = config.config.chat_queue_url
        self.region_name = config.config.aws_region
        self.endpoint_url = config.config.localstack_url
        self._client = None
        self._resolved_queue_url = None

    async def __aenter__(self):
        """Create async SQS client on context manager entry."""
        self._client = await self.session.create_client(
            "sqs",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        ).__aenter__()

        # Use configured queue URL directly
        # LocalStack will handle the internal routing
        self._resolved_queue_url = self.queue_url
        logger.info("Using queue URL: %s", self._resolved_queue_url)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async SQS client on context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def send_message(self, message_body: dict) -> str:
        """Send a message to the SQS queue."""
        response = await self._client.send_message(
            QueueUrl=self._resolved_queue_url, MessageBody=json.dumps(message_body)
        )
        return response["MessageId"]

    async def receive_messages(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        wait_time: int = DEFAULT_LONG_POLL_WAIT_SECONDS,
    ) -> list:
        """Receive messages from the SQS queue with long polling."""
        response = await self._client.receive_message(
            QueueUrl=self._resolved_queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
        )
        return response.get("Messages", [])

    async def delete_message(self, receipt_handle: str) -> None:
        """Delete a message from the SQS queue."""
        await self._client.delete_message(
            QueueUrl=self._resolved_queue_url, ReceiptHandle=receipt_handle
        )
