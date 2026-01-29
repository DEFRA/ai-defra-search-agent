import asyncio
import json
import logging

import boto3

from app import config

logger = logging.getLogger(__name__)

# SQS default polling configuration
DEFAULT_MAX_MESSAGES = 1
DEFAULT_LONG_POLL_WAIT_SECONDS = 20


class SQSClient:
    """Async wrapper around synchronous boto3 SQS client.

    Uses `asyncio.to_thread` to avoid blocking the event loop while calling
    boto3's synchronous methods. This keeps the async API (`async with` and
    `await send_message`) while using the standard boto3 client.
    """

    def __init__(self):
        self.queue_url = config.config.sqs_chat_queue_url
        self.region_name = config.config.aws_region
        self.endpoint_url = config.config.localstack_url
        self._client = None
        self._resolved_queue_url = None

    async def __aenter__(self):
        """Create boto3 SQS client on async context manager entry.

        Creating the client is fast and synchronous; we do it directly and
        wrap long-running operations (send/receive/delete) in threads.
        """
        # Create synchronous boto3 client
        self._client = boto3.client(
            "sqs", region_name=self.region_name, endpoint_url=self.endpoint_url
        )

        # Use configured queue URL directly (LocalStack handles routing)
        self._resolved_queue_url = self.queue_url
        logger.info("Using queue URL: %s", self._resolved_queue_url)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the synchronous boto3 client without blocking the loop."""
        if self._client:
            await asyncio.to_thread(self._client.close)

    async def send_message(self, message_body: dict) -> str:
        """Send a message to the SQS queue using a thread to avoid blocking."""

        def _send():
            return self._client.send_message(
                QueueUrl=self._resolved_queue_url, MessageBody=json.dumps(message_body)
            )

        response = await asyncio.to_thread(_send)
        return response.get("MessageId")

    async def receive_messages(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        wait_time: int = DEFAULT_LONG_POLL_WAIT_SECONDS,
    ) -> list:
        """Receive messages from the SQS queue with long polling.

        Long-polling is blocking in boto3; this runs in a thread so the event
        loop remains responsive. For high throughput consider using a separate
        worker process instead.
        """

        def _receive():
            return self._client.receive_message(
                QueueUrl=self._resolved_queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
            )

        response = await asyncio.to_thread(_receive)
        return response.get("Messages", [])

    async def delete_message(self, receipt_handle: str) -> None:
        """Delete a message from the SQS queue using a thread."""

        def _delete():
            return self._client.delete_message(
                QueueUrl=self._resolved_queue_url, ReceiptHandle=receipt_handle
            )

        await asyncio.to_thread(_delete)
