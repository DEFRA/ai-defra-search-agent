import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError

from app.chat import models, worker


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    chat_service = AsyncMock()
    conversation_repository = AsyncMock()
    sqs_client = AsyncMock()
    return chat_service, conversation_repository, sqs_client


@pytest.fixture
def sample_message():
    """Create a sample SQS message."""
    message_id = str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    return {
        "Body": json.dumps(
            {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "question": "What is AI?",
                "model_id": "anthropic.claude-3-haiku",
            }
        ),
        "ReceiptHandle": "test-receipt-handle",
    }


@pytest.mark.asyncio
async def test_process_job_success(mock_services, sample_message):
    """Test successful message processing."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock conversation response
    mock_conversation = MagicMock()
    mock_conversation.id = uuid.uuid4()
    mock_message = MagicMock()
    mock_message.message_id = uuid.uuid4()
    mock_message.role = "user"
    mock_message.content = "What is AI?"
    mock_message.model_name = "Claude Sonnet 3.7"
    mock_message.model_id = "anthropic.claude-3-haiku"
    mock_message.status = models.MessageStatus.COMPLETED
    mock_message.timestamp = MagicMock()
    mock_message.timestamp.isoformat.return_value = "2024-01-01T00:00:00"

    mock_conversation.messages = [mock_message]
    chat_service.execute_chat.return_value = mock_conversation

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify message status updates
    body = json.loads(sample_message["Body"])
    message_id = uuid.UUID(body["message_id"])
    conversation_id = uuid.UUID(body["conversation_id"])

    # Should update to PROCESSING first
    assert conversation_repository.update_message_status.call_count == 2
    first_call = conversation_repository.update_message_status.call_args_list[0]
    assert first_call[1]["conversation_id"] == conversation_id
    assert first_call[1]["message_id"] == message_id
    assert first_call[1]["status"] == models.MessageStatus.PROCESSING

    # Then to COMPLETED
    second_call = conversation_repository.update_message_status.call_args_list[1]
    assert second_call[1]["status"] == models.MessageStatus.COMPLETED

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once_with("test-receipt-handle")


@pytest.mark.asyncio
async def test_run_worker_polls_and_processes(monkeypatch):
    """Run a short-lived worker loop and assert it polls and calls process_job_message."""
    import asyncio as _asyncio
    import contextlib as _contextlib
    import json as _json

    from app.chat import dependencies as deps_mod
    from app.chat import worker as worker_mod

    evt = _asyncio.Event()

    class DummySQS:
        def __init__(self):
            self.receive_calls = 0
            self.deleted = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def receive_messages(self, max_messages=None, wait_time=None):
            # reference the params so linters don't mark them unused
            _ = max_messages
            _ = wait_time
            self.receive_calls += 1
            # signal test that we've received at least one poll
            evt.set()
            if self.receive_calls == 1:
                return [
                    {
                        "Body": _json.dumps(
                            {
                                "message_id": str(uuid.uuid4()),
                                "conversation_id": str(uuid.uuid4()),
                                "question": "q",
                                "model_id": "m1",
                            }
                        ),
                        "ReceiptHandle": "rh",
                    }
                ]
            # On subsequent calls, raise to trigger the exception path
            msg = "stop"
            raise Exception(msg)

        async def delete_message(self, receipt_handle):
            self.deleted.append(receipt_handle)

    dummy = DummySQS()
    chat_service = AsyncMock()
    conv_repo = AsyncMock()

    monkeypatch.setattr(
        deps_mod,
        "initialize_worker_services",
        AsyncMock(return_value=(chat_service, conv_repo, dummy)),
    )
    proc = AsyncMock()
    monkeypatch.setattr(worker_mod, "process_job_message", proc)

    # Speed up loops
    monkeypatch.setattr(worker_mod, "SQS_LONG_POLL_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "WORKER_ERROR_RETRY_DELAY_SECONDS", 0.01)

    task = _asyncio.create_task(worker_mod.run_worker())
    # Wait deterministically until first receive happens
    await _asyncio.wait_for(evt.wait(), timeout=1.0)
    task.cancel()
    with _contextlib.suppress(_asyncio.CancelledError):
        await task

    # Assertions
    assert proc.await_count >= 1
    assert dummy.receive_calls >= 1
    assert worker_mod.get_last_heartbeat() is not None


def test_update_message_failed_no_conversation():
    """_update_message_failed should be a no-op when conversation_id is None."""
    from unittest.mock import Mock

    from app.chat import worker as worker_mod

    repo = Mock()
    # call the internal helper with no conversation id
    import asyncio

    asyncio.run(worker_mod._update_message_failed(repo, None, uuid.uuid4(), "err", 500))
    # repo.update_message_status should not have been called
    assert not repo.update_message_status.called


def test_get_last_heartbeat_roundtrip():
    # setting private variable and reading it back
    import datetime

    from app.chat import worker as worker_mod

    now = datetime.datetime.now(datetime.UTC)
    worker_mod._last_heartbeat = now
    assert worker_mod.get_last_heartbeat() == now


@pytest.mark.asyncio
async def test_process_job_conversation_not_found(mock_services, sample_message):
    """Test handling of conversation not found error."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise ConversationNotFoundError
    chat_service.execute_chat.side_effect = models.ConversationNotFoundError(
        "Conversation not found"
    )

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify message status updated to FAILED
    body = json.loads(sample_message["Body"])
    message_id = uuid.UUID(body["message_id"])
    conversation_id = uuid.UUID(body["conversation_id"])

    # Should update to PROCESSING, then FAILED
    assert conversation_repository.update_message_status.call_count == 2
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["conversation_id"] == conversation_id
    assert failed_call[1]["message_id"] == message_id
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert "Conversation not found" in failed_call[1]["error_message"]
    assert failed_call[1]["error_code"] == 404

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_throttling_exception(mock_services, sample_message):
    """Test handling of AWS throttling exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise ClientError with ThrottlingException
    error_response = {
        "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify error code mapping
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert failed_call[1]["error_code"] == 429  # HTTP_TOO_MANY_REQUESTS

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_service_unavailable_exception(mock_services, sample_message):
    """Test handling of AWS service unavailable exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise ClientError with ServiceUnavailableException
    error_response = {
        "Error": {
            "Code": "ServiceUnavailableException",
            "Message": "Service unavailable",
        },
        "ResponseMetadata": {"HTTPStatusCode": 503},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify error code mapping
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert failed_call[1]["error_code"] == 503  # HTTP_SERVICE_UNAVAILABLE

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_internal_server_exception(mock_services, sample_message):
    """Test handling of AWS internal server exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise ClientError with InternalServerException
    error_response = {
        "Error": {"Code": "InternalServerException", "Message": "Internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify error code mapping
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert failed_call[1]["error_code"] == 500  # HTTP_INTERNAL_SERVER_ERROR

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_generic_client_error(mock_services, sample_message):
    """Test handling of generic AWS client error."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise ClientError with unknown error code
    error_response = {
        "Error": {"Code": "UnknownError", "Message": "Unknown error"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify error preserved from response
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert "UnknownError" in failed_call[1]["error_message"]

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_generic_exception(mock_services, sample_message):
    """Test handling of generic exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    # Setup mock to raise generic exception
    chat_service.execute_chat.side_effect = Exception("Test error")

    # Execute
    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify error handling
    failed_call = conversation_repository.update_message_status.call_args_list[1]
    assert failed_call[1]["status"] == models.MessageStatus.FAILED
    assert failed_call[1]["error_message"] == "Test error"
    assert failed_call[1]["error_code"] == 500

    # Verify message deletion
    sqs_client.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_deletes_message_on_exception(mock_services, sample_message):
    """Test that SQS message is always deleted even when an exception occurs."""
    chat_service, conversation_repository, sqs_client = mock_services

    chat_service.execute_chat.side_effect = Exception("Test error")

    await worker.process_job_message(
        sample_message, chat_service, conversation_repository, sqs_client
    )

    # Verify message was deleted despite exception
    sqs_client.delete_message.assert_awaited_once_with("test-receipt-handle")
