import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.chat import models, worker


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    chat_service = AsyncMock()
    conversation_repository = AsyncMock()
    sqs_client = MagicMock()

    async def _mock_claim(conversation_id, message_id):
        await conversation_repository.update_message_status(
            conversation_id=conversation_id,
            message_id=message_id,
            status=models.MessageStatus.PROCESSING,
        )
        return True

    conversation_repository.claim_message = AsyncMock(side_effect=_mock_claim)

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

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        body = json.loads(sample_message["Body"])
        message_id = uuid.UUID(body["message_id"])
        conversation_id = uuid.UUID(body["conversation_id"])

        assert conversation_repository.update_message_status.call_count == 2
        first_call = conversation_repository.update_message_status.call_args_list[0]
        assert first_call[1]["conversation_id"] == conversation_id
        assert first_call[1]["message_id"] == message_id
        assert first_call[1]["status"] == models.MessageStatus.PROCESSING

        second_call = conversation_repository.update_message_status.call_args_list[1]
        assert second_call[1]["status"] == models.MessageStatus.COMPLETED

        mock_to_thread.assert_called_once_with(
            sqs_client.delete_message, "test-receipt-handle"
        )


@pytest.mark.asyncio
async def test_run_worker_polls_and_processes(monkeypatch):
    """Run a short-lived worker loop and assert it polls and calls process_job_message."""
    import asyncio as _asyncio
    import contextlib as _contextlib
    import json as _json

    from app import config as config_mod
    from app.chat import dependencies as deps_mod
    from app.chat import worker as worker_mod

    evt = _asyncio.Event()
    proc_called = _asyncio.Event()

    class ChatQueueCfg:
        wait_time = 0.01
        polling_interval = 0.01
        batch_size = 1

    class MockConfig:
        chat_queue = ChatQueueCfg()

    class DummySQS:
        def __init__(self):
            self.receive_calls = 0
            self.deleted = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def receive_messages(self, max_messages=None, wait_time=None):
            _ = max_messages
            _ = wait_time
            self.receive_calls += 1
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
            msg = "stop"
            raise Exception(msg)

        def delete_message(self, receipt_handle):
            self.deleted.append(receipt_handle)

    dummy = DummySQS()
    chat_service = AsyncMock()
    conv_repo = AsyncMock()

    monkeypatch.setattr(config_mod, "config", MockConfig())
    monkeypatch.setattr(
        deps_mod,
        "initialize_worker_services",
        AsyncMock(return_value=(chat_service, conv_repo, dummy)),
    )

    async def mock_process(*_args, **_kwargs):
        proc_called.set()

    proc = AsyncMock(side_effect=mock_process)
    monkeypatch.setattr(worker_mod, "process_job_message", proc)

    task = _asyncio.create_task(worker_mod.run_worker())
    await _asyncio.wait_for(evt.wait(), timeout=1.0)

    with _contextlib.suppress(TimeoutError):
        await _asyncio.wait_for(proc_called.wait(), timeout=0.5)

    task.cancel()
    with _contextlib.suppress(_asyncio.CancelledError):
        await task

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

    asyncio.run(worker_mod._update_message_failed(repo, None, uuid.uuid4(), "err"))
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

    chat_service.execute_chat.side_effect = models.ConversationNotFoundError(
        "Conversation not found"
    )

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        body = json.loads(sample_message["Body"])
        message_id = uuid.UUID(body["message_id"])
        conversation_id = uuid.UUID(body["conversation_id"])

        assert conversation_repository.update_message_status.call_count == 2
        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["conversation_id"] == conversation_id
        assert failed_call[1]["message_id"] == message_id
        assert failed_call[1]["status"] == models.MessageStatus.FAILED
        assert "Conversation not found" in failed_call[1]["error_message"]

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_throttling_exception(mock_services, sample_message):
    """Test handling of AWS throttling exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    error_response = {
        "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["status"] == models.MessageStatus.FAILED

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_service_unavailable_exception(mock_services, sample_message):
    """Test handling of AWS service unavailable exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    error_response = {
        "Error": {
            "Code": "ServiceUnavailableException",
            "Message": "Service unavailable",
        },
        "ResponseMetadata": {"HTTPStatusCode": 503},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["status"] == models.MessageStatus.FAILED

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_internal_server_exception(mock_services, sample_message):
    """Test handling of AWS internal server exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    error_response = {
        "Error": {"Code": "InternalServerException", "Message": "Internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["status"] == models.MessageStatus.FAILED

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_generic_client_error(mock_services, sample_message):
    """Test handling of generic AWS client error."""
    chat_service, conversation_repository, sqs_client = mock_services

    error_response = {
        "Error": {"Code": "UnknownError", "Message": "Unknown error"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    chat_service.execute_chat.side_effect = ClientError(error_response, "InvokeModel")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["status"] == models.MessageStatus.FAILED
        assert "UnknownError" in failed_call[1]["error_message"]

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_generic_exception(mock_services, sample_message):
    """Test handling of generic exception."""
    chat_service, conversation_repository, sqs_client = mock_services

    chat_service.execute_chat.side_effect = Exception("Test error")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        failed_call = conversation_repository.update_message_status.call_args_list[1]
        assert failed_call[1]["status"] == models.MessageStatus.FAILED
        assert failed_call[1]["error_message"] == "Test error"

        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_process_job_deletes_message_on_exception(mock_services, sample_message):
    """Test that SQS message is always deleted even when an exception occurs."""
    chat_service, conversation_repository, sqs_client = mock_services

    chat_service.execute_chat.side_effect = Exception("Test error")

    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = None

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        mock_to_thread.assert_called_once_with(
            sqs_client.delete_message, "test-receipt-handle"
        )
