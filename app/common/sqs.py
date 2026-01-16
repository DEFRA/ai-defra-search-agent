import json

import boto3

from app import config


class SQSClient:
    def __init__(self):
        self.client = boto3.client(
            "sqs",
            region_name=config.config.aws_region,
            endpoint_url=config.config.localstack_url,
        )
        self.queue_url = config.config.chat_queue_url

    def send_message(self, message_body: dict) -> str:
        response = self.client.send_message(
            QueueUrl=self.queue_url, MessageBody=json.dumps(message_body)
        )
        return response["MessageId"]

    def receive_messages(self, max_messages: int = 1, wait_time: int = 20) -> list:
        response = self.client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
        )
        return response.get("Messages", [])

    def delete_message(self, receipt_handle: str) -> None:
        self.client.delete_message(
            QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
        )
