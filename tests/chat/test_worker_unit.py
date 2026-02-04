import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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
    chat_service = AsyncMock()
    chat_service.execute_chat.return_value = models.Conversation()

    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            msg, chat_service, conversation_repository, sqs_client
        )

        assert conversation_repository.update_message_status.await_count >= 1
        mock_to_thread.assert_called_once_with(sqs_client.delete_message, "rh")


@pytest.mark.asyncio
async def test_process_job_message_conversation_not_found():
    chat_service = AsyncMock()
    chat_service.execute_chat.side_effect = models.ConversationNotFoundError("missing")

    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            msg, chat_service, conversation_repository, sqs_client
        )

        assert conversation_repository.update_message_status.await_count >= 1
        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_message_client_error():
    chat_service = AsyncMock()
    response = {
        "Error": {"Code": "ThrottlingException", "Message": "throttle"},
        "ResponseMetadata": {"HTTPStatusCode": 429},
    }
    err = ClientError(response, "Invoke")
    chat_service.execute_chat.side_effect = err

    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            msg, chat_service, conversation_repository, sqs_client
        )

        assert conversation_repository.update_message_status.await_count >= 1
        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_message_client_error_mapping_detailed():
    chat_service = AsyncMock()
    response = {
        "Error": {"Code": "ThrottlingException", "Message": "throttle"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    err = ClientError(response, "Invoke")
    chat_service.execute_chat.side_effect = err

    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            msg, chat_service, conversation_repository, sqs_client
        )

        found = False
        for call in conversation_repository.update_message_status.await_args_list:
            if call.kwargs.get("error_code") == 429:
                found = True
            if len(call.args) > 4 and call.args[4] == 429:
                found = True
        assert found


@pytest.mark.asyncio
async def test_process_job_message_general_exception_calls_delete():
    chat_service = AsyncMock()
    chat_service.execute_chat.side_effect = ValueError("boom")

    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            msg, chat_service, conversation_repository, sqs_client
        )

        mock_to_thread.assert_called_once()
