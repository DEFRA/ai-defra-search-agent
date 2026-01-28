import boto3
import pytest

from app.common import sqs


class DummyBotoClient:
    def __init__(self):
        self._messages = []

    def send_message(self, *_, **__):
        return {"MessageId": "msg-123"}

    def receive_message(self, *_, **__):
        return {"Messages": [{"Body": "{}", "ReceiptHandle": "rh"}]}

    def delete_message(self, *_, **__):
        return {}

    def close(self):
        return None


@pytest.mark.asyncio
async def test_sqs_client_send_receive_delete(monkeypatch):
    dummy = DummyBotoClient()

    # Patch boto3.client to return our dummy
    def _fake_boto_client(*_args, **_kwargs):
        return dummy

    monkeypatch.setattr(boto3, "client", _fake_boto_client)

    # Provide a fake config so client picks up queue and endpoint values
    class Cfg:
        chat_queue_url = "http://example"
        aws_region = "eu-west-2"
        localstack_url = None

    from app import config

    old = config.config
    config.config = Cfg()

    try:
        async with sqs.SQSClient() as client:
            msg_id = await client.send_message({"hello": "world"})
            assert msg_id == "msg-123"

            messages = await client.receive_messages(max_messages=1, wait_time=0)
            assert isinstance(messages, list)

            await client.delete_message("rh")
    finally:
        config.config = old
