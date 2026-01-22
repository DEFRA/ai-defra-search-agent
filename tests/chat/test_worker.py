import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError

from app.chat import models, worker


class TestProcessJobMessage:
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        chat_service = AsyncMock()
        conversation_repository = AsyncMock()
        sqs_client = AsyncMock()
        return chat_service, conversation_repository, sqs_client

    @pytest.fixture
    def sample_message(self):
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
    async def test_process_job_success(self, mock_services, sample_message, mocker):
        """Test successful message processing."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

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

        # Verify event broker publishes
        assert mock_event_broker.publish.call_count == 2

        # Verify message deletion
        sqs_client.delete_message.assert_awaited_once_with("test-receipt-handle")

    @pytest.mark.asyncio
    async def test_process_job_conversation_not_found(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of conversation not found error."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

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
    async def test_process_job_throttling_exception(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of AWS throttling exception."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

        # Setup mock to raise ClientError with ThrottlingException
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

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
    async def test_process_job_service_unavailable_exception(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of AWS service unavailable exception."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

        # Setup mock to raise ClientError with ServiceUnavailableException
        error_response = {
            "Error": {
                "Code": "ServiceUnavailableException",
                "Message": "Service unavailable",
            },
            "ResponseMetadata": {"HTTPStatusCode": 503},
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

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
    async def test_process_job_internal_server_exception(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of AWS internal server exception."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

        # Setup mock to raise ClientError with InternalServerException
        error_response = {
            "Error": {"Code": "InternalServerException", "Message": "Internal error"},
            "ResponseMetadata": {"HTTPStatusCode": 500},
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

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
    async def test_process_job_generic_client_error(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of generic AWS client error."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

        # Setup mock to raise ClientError with unknown error code
        error_response = {
            "Error": {"Code": "UnknownError", "Message": "Unknown error"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

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
    async def test_process_job_generic_exception(
        self, mock_services, sample_message, mocker
    ):
        """Test handling of generic exception."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

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
    async def test_process_job_deletes_message_on_exception(
        self, mock_services, sample_message, mocker
    ):
        """Test that SQS message is always deleted even when an exception occurs."""
        chat_service, conversation_repository, sqs_client = mock_services

        # Mock event broker
        mock_event_broker = AsyncMock()
        mocker.patch("app.chat.worker.get_event_broker", return_value=mock_event_broker)

        chat_service.execute_chat.side_effect = Exception("Test error")

        await worker.process_job_message(
            sample_message, chat_service, conversation_repository, sqs_client
        )

        # Verify message was deleted despite exception
        sqs_client.delete_message.assert_awaited_once_with("test-receipt-handle")
