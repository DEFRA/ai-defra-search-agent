import json
import uuid
from unittest.mock import AsyncMock

import pytest

from app.chat import models, worker
from app.chat.models import Conversation


@pytest.mark.asyncio
async def test_claim_success_calls_execute_and_updates_and_deletes():
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    body = {
        "conversation_id": str(conversation_id),
        "message_id": str(message_id),
        "question": "hello",
        "model_id": "m1",
    }
    message = {"Body": json.dumps(body), "ReceiptHandle": "rh-1"}

    chat_service = AsyncMock()
    conv = Conversation(id=conversation_id)
    chat_service.execute_chat = AsyncMock(return_value=conv)

    conv_repo = AsyncMock()
    conv_repo.claim_message = AsyncMock(return_value=True)
    conv_repo.update_message_status = AsyncMock()

    sqs_client = AsyncMock()
    sqs_client.delete_message = AsyncMock()

    await worker.process_job_message(message, chat_service, conv_repo, sqs_client)

    chat_service.execute_chat.assert_awaited_once()
    conv_repo.update_message_status.assert_awaited_once_with(
        conversation_id=conv.id,
        message_id=message_id,
        status=models.MessageStatus.COMPLETED,
    )
    sqs_client.delete_message.assert_awaited_once_with("rh-1")


@pytest.mark.asyncio
async def test_claim_failed_and_completed_status_acknowledges_and_skips():
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    body = {
        "conversation_id": str(conversation_id),
        "message_id": str(message_id),
        "question": "hello",
        "model_id": "m1",
    }
    message = {"Body": json.dumps(body), "ReceiptHandle": "rh-2"}

    chat_service = AsyncMock()
    chat_service.execute_chat = AsyncMock()

    conv_repo = AsyncMock()
    conv_repo.claim_message = AsyncMock(return_value=False)
    conv_repo.get_message_status = AsyncMock(
        return_value=models.MessageStatus.COMPLETED
    )
    conv_repo.update_message_status = AsyncMock()

    sqs_client = AsyncMock()
    sqs_client.delete_message = AsyncMock()

    await worker.process_job_message(message, chat_service, conv_repo, sqs_client)

    chat_service.execute_chat.assert_not_awaited()
    conv_repo.update_message_status.assert_not_awaited()
    sqs_client.delete_message.assert_awaited_once_with("rh-2")


@pytest.mark.asyncio
async def test_claim_failed_and_processing_status_acknowledges_and_skips():
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    body = {
        "conversation_id": str(conversation_id),
        "message_id": str(message_id),
        "question": "hello",
        "model_id": "m1",
    }
    message = {"Body": json.dumps(body), "ReceiptHandle": "rh-3"}

    chat_service = AsyncMock()
    chat_service.execute_chat = AsyncMock()

    conv_repo = AsyncMock()
    conv_repo.claim_message = AsyncMock(return_value=False)
    conv_repo.get_message_status = AsyncMock(
        return_value=models.MessageStatus.PROCESSING
    )
    conv_repo.update_message_status = AsyncMock()

    sqs_client = AsyncMock()
    sqs_client.delete_message = AsyncMock()

    await worker.process_job_message(message, chat_service, conv_repo, sqs_client)

    chat_service.execute_chat.assert_not_awaited()
    conv_repo.update_message_status.assert_not_awaited()
    sqs_client.delete_message.assert_awaited_once_with("rh-3")


@pytest.mark.asyncio
async def test_claim_failed_and_missing_record_acknowledges_and_skips():
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()
    body = {
        "conversation_id": str(conversation_id),
        "message_id": str(message_id),
        "question": "hello",
        "model_id": "m1",
    }
    message = {"Body": json.dumps(body), "ReceiptHandle": "rh-4"}

    chat_service = AsyncMock()
    chat_service.execute_chat = AsyncMock()

    conv_repo = AsyncMock()
    conv_repo.claim_message = AsyncMock(return_value=False)
    conv_repo.get_message_status = AsyncMock(return_value=None)
    conv_repo.update_message_status = AsyncMock()

    sqs_client = AsyncMock()
    sqs_client.delete_message = AsyncMock()

    await worker.process_job_message(message, chat_service, conv_repo, sqs_client)

    chat_service.execute_chat.assert_not_awaited()
    conv_repo.update_message_status.assert_not_awaited()
    sqs_client.delete_message.assert_awaited_once_with("rh-4")
