import json
import uuid

import pytest
from botocore.exceptions import ClientError
from pytest_mock import MockerFixture

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
async def test_process_job_message_success(mocker: MockerFixture):
    chat_service = mocker.AsyncMock()
    chat_service.execute_chat.return_value = models.Conversation()

    conversation_repository = mocker.AsyncMock()
    sqs_client = mocker.MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_to_thread.return_value = None

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    assert conversation_repository.update_message_status.await_count >= 1
    mock_to_thread.assert_called_once_with(sqs_client.delete_message, "rh")


@pytest.mark.asyncio
async def test_process_job_message_conversation_not_found(mocker: MockerFixture):
    chat_service = mocker.AsyncMock()
    chat_service.execute_chat.side_effect = models.ConversationNotFoundError("missing")

    conversation_repository = mocker.AsyncMock()
    sqs_client = mocker.MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_to_thread.return_value = None

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    assert conversation_repository.update_message_status.await_count >= 1
    mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_message_client_error(mocker: MockerFixture):
    chat_service = mocker.AsyncMock()
    response = {
        "Error": {"Code": "ThrottlingException", "Message": "throttle"},
        "ResponseMetadata": {"HTTPStatusCode": 429},
    }
    err = ClientError(response, "Invoke")
    chat_service.execute_chat.side_effect = err

    conversation_repository = mocker.AsyncMock()
    sqs_client = mocker.MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_to_thread.return_value = None

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    assert conversation_repository.update_message_status.await_count >= 1
    mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_message_client_error_mapping_detailed(mocker: MockerFixture):
    chat_service = mocker.AsyncMock()
    response = {
        "Error": {"Code": "ThrottlingException", "Message": "throttle"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    err = ClientError(response, "Invoke")
    chat_service.execute_chat.side_effect = err

    conversation_repository = mocker.AsyncMock()
    sqs_client = mocker.MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_to_thread.return_value = None

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    # Verify that update_message_status was called with FAILED status
    found = False
    for call in conversation_repository.update_message_status.await_args_list:
        if call.kwargs.get("status") == models.MessageStatus.FAILED:
            found = True
    assert found


@pytest.mark.asyncio
async def test_process_job_message_general_exception_calls_delete(
    mocker: MockerFixture,
):
    chat_service = mocker.AsyncMock()
    chat_service.execute_chat.side_effect = ValueError("boom")

    conversation_repository = mocker.AsyncMock()
    sqs_client = mocker.MagicMock()

    msg = make_message_body(conversation_id=uuid.uuid4())

    mock_to_thread = mocker.patch("asyncio.to_thread")
    mock_to_thread.return_value = None

    await worker.process_job_message(
        msg, chat_service, conversation_repository, sqs_client
    )

    mock_to_thread.assert_called_once()
