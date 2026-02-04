"""Simple wrapper around boto3 SQS client.

This module exposes `SQSClient`, a lightweight wrapper around boto3's
SQS client. It provides `send_message`, `receive_messages` and `delete_message`
operations. The worker is responsible for handling these synchronous operations
appropriately in its execution context.
"""

import logging

import boto3

from app import config

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 1
DEFAULT_LONG_POLL_WAIT_SECONDS = 20


class SQSClient:
    """Synchronous boto3 SQS client wrapper."""

    def __init__(self):
        self.queue_url = config.config.sqs_chat_queue_url
        self.region_name = config.config.aws_region
        self.endpoint_url = config.config.localstack_url
        self._client = None
        self._resolved_queue_url = None

    def __enter__(self):
        """Create the underlying boto3 SQS client."""
        self._client = boto3.client(
            "sqs", region_name=self.region_name, endpoint_url=self.endpoint_url
        )

        self._resolved_queue_url = self.queue_url
        logger.debug("Using queue URL: %s", self._resolved_queue_url)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the boto3 client."""
        if self._client:
            self._client.close()

    def send_message(self, message_body: str) -> str:
        """Send a raw string message body to the configured SQS queue.

        Args:
            message_body: Raw string to send as the SQS `MessageBody`.

        Returns:
            str: The SQS `MessageId` assigned by the queue.
        """
        response = self._client.send_message(
            QueueUrl=self._resolved_queue_url, MessageBody=message_body
        )
        return response.get("MessageId")

    def receive_messages(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        wait_time: int = DEFAULT_LONG_POLL_WAIT_SECONDS,
    ) -> list:
        """Long-poll the configured SQS queue and return any messages.

        Args:
            max_messages: Maximum messages to request in a single poll.
            wait_time: Long poll wait time in seconds.

        Returns:
            list: Sequence of message dicts, or an empty list.
        """
        response = self._client.receive_message(
            QueueUrl=self._resolved_queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
        )
        return response.get("Messages", [])

    def delete_message(self, receipt_handle: str) -> None:
        """Delete a message from SQS using its `ReceiptHandle`.

        Args:
            receipt_handle: The SQS receipt handle for the message to delete.
        """
        self._client.delete_message(
            QueueUrl=self._resolved_queue_url, ReceiptHandle=receipt_handle
        )
