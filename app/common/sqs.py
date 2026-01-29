"""Asynchronous helper wrapper around boto3 SQS client.

This module exposes `SQSClient`, a small async-friendly wrapper that
delegates blocking boto3 calls to threads via `asyncio.to_thread`. It
provides simple `send_message`, `receive_messages` and `delete_message`
operations and supports use as an async context manager.
"""

import asyncio
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
        """Create the underlying boto3 SQS client.

        The boto3 client is synchronous; long-running operations are executed
        in a thread to avoid blocking the event loop. The resolved queue URL
        is loaded from the runtime configuration (which may point to
        LocalStack in development).
        """
        # Create synchronous boto3 client
        self._client = boto3.client(
            "sqs", region_name=self.region_name, endpoint_url=self.endpoint_url
        )

        # Use configured queue URL directly (LocalStack handles routing)
        self._resolved_queue_url = self.queue_url
        logger.debug("Using queue URL: %s", self._resolved_queue_url)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the synchronous boto3 client without blocking the loop."""
        if self._client:
            await asyncio.to_thread(self._client.close)

    async def send_message(self, message_body: str) -> str:
        """Send a raw string message body to the configured SQS queue.

        Args:
            message_body: Raw string to send as the SQS `MessageBody`.

        Returns:
            str: The SQS `MessageId` assigned by the queue.
        """

        def _send():
            return self._client.send_message(
                QueueUrl=self._resolved_queue_url, MessageBody=message_body
            )

        response = await asyncio.to_thread(_send)
        return response.get("MessageId")

    async def receive_messages(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        wait_time: int = DEFAULT_LONG_POLL_WAIT_SECONDS,
    ) -> list:
        """Long-poll the configured SQS queue and return any messages.

        Long-polling is blocking in boto3; this implementation runs the
        underlying call in a thread so the async event loop remains
        responsive. The returned messages are raw boto3 dictionary objects as
        returned by `receive_message`.

        Args:
            max_messages: Maximum messages to request in a single poll.
            wait_time: Long poll wait time in seconds.

        Returns:
            list: Sequence of message dicts, or an empty list.
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
        """Delete a message from SQS using its `ReceiptHandle`.

        Args:
            receipt_handle: The SQS receipt handle for the message to delete.
        """

        def _delete():
            return self._client.delete_message(
                QueueUrl=self._resolved_queue_url, ReceiptHandle=receipt_handle
            )

        await asyncio.to_thread(_delete)
