import json
import uuid
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from app.chat import models, worker


def make_message_body(conversation_id=None, message_id=None):
    return {
        "Body": json.dumps(
            {
                "conversation_id": str(conversation_id) if conversation_id else None,
                "message_id": str(message_id or uuid.uuid4()),
                "question": "hello",
                "model_id": "mid",
            }
        ),
        "ReceiptHandle": "rh",
    }


@pytest.mark.asyncio
async def test_process_job_message_success():
    # Setup mocks
    chat_service = AsyncMock()
    # chat_service.execute_chat returns a Conversation
    chat_service.execute_chat.return_value = models.Conversation()

    conversation_repository = AsyncMock()
    sqs_client = AsyncMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    # ensure processing steps called and delete_message called
    assert conversation_repository.update_message_status.await_count >= 1
    sqs_client.delete_message.assert_awaited_once_with("rh")


@pytest.mark.asyncio
async def test_process_job_message_conversation_not_found():
    chat_service = AsyncMock()
    # make execute_chat raise ConversationNotFoundError
    chat_service.execute_chat.side_effect = models.ConversationNotFoundError("missing")

    conversation_repository = AsyncMock()
    sqs_client = AsyncMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    # should mark message failed and delete message
    assert conversation_repository.update_message_status.await_count >= 1
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_message_client_error():
    chat_service = AsyncMock()
    # simulate boto3 ClientError
    response = {
        "Error": {"Code": "ThrottlingException", "Message": "throttle"},
        "ResponseMetadata": {"HTTPStatusCode": 429},
    }
    err = ClientError(response, "Invoke")
    chat_service.execute_chat.side_effect = err

    conversation_repository = AsyncMock()
    sqs_client = AsyncMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    # should mark message failed and delete message
    assert conversation_repository.update_message_status.await_count >= 1
    sqs_client.delete_message.assert_awaited_once()
