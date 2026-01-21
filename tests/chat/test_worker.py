import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.chat import job_models, models, worker


class TestProcessJobMessage:
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        chat_service = AsyncMock()
        job_repository = AsyncMock()
        sqs_client = AsyncMock()
        return chat_service, job_repository, sqs_client

    @pytest.fixture
    def sample_message(self):
        """Create a sample SQS message."""
        job_id = str(uuid.uuid4())
        return {
            "Body": json.dumps({
                "job_id": job_id,
                "question": "What is AI?",
                "model_id": "anthropic.claude-3-haiku",
                "conversation_id": None
            }),
            "ReceiptHandle": "test-receipt-handle"
        }

    @pytest.mark.asyncio
    async def test_process_job_success(self, mock_services, sample_message):
        """Test successful job processing."""
        chat_service, job_repository, sqs_client = mock_services

        # Setup mock conversation response
        mock_conversation = MagicMock()
        mock_conversation.id = uuid.uuid4()
        mock_message_user = MagicMock()
        mock_message_user.role = "user"
        mock_message_user.content = "What is AI?"
        mock_message_user.model_name = "Claude Sonnet 3.7"
        mock_message_user.model_id = "anthropic.claude-3-haiku"
        mock_message_user.timestamp = MagicMock()
        mock_message_user.timestamp.isoformat.return_value = "2024-01-01T00:00:00"

        mock_message_assistant = MagicMock()
        mock_message_assistant.role = "assistant"
        mock_message_assistant.content = "AI stands for Artificial Intelligence"
        mock_message_assistant.model_name = "Claude Sonnet 3.7"
        mock_message_assistant.model_id = "anthropic.claude-3-haiku"
        mock_message_assistant.timestamp = MagicMock()
        mock_message_assistant.timestamp.isoformat.return_value = "2024-01-01T00:00:01"

        mock_conversation.messages = [mock_message_user, mock_message_assistant]
        chat_service.execute_chat.return_value = mock_conversation

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        body = json.loads(sample_message["Body"])
        job_id = body["job_id"]

        # Verify job was updated to PROCESSING
        job_repository.update.assert_any_call(
            job_id=job_id, status=job_models.JobStatus.PROCESSING
        )

        # Verify chat service was called
        chat_service.execute_chat.assert_called_once_with(
            question="What is AI?",
            model_id="anthropic.claude-3-haiku",
            conversation_id=None
        )

        # Verify job was updated to COMPLETED with result
        completed_call = [call for call in job_repository.update.call_args_list 
                         if call.kwargs.get("status") == job_models.JobStatus.COMPLETED][0]
        assert completed_call.kwargs["result"]["conversation_id"] == str(mock_conversation.id)
        assert len(completed_call.kwargs["result"]["messages"]) == 2

        # Verify message was deleted from SQS
        sqs_client.delete_message.assert_called_once_with("test-receipt-handle")

    @pytest.mark.asyncio
    async def test_process_job_conversation_not_found(self, mock_services, sample_message):
        """Test handling of ConversationNotFoundError."""
        chat_service, job_repository, sqs_client = mock_services

        chat_service.execute_chat.side_effect = models.ConversationNotFoundError(
            "Conversation not found"
        )

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        body = json.loads(sample_message["Body"])
        job_id = body["job_id"]

        # Verify job was marked as FAILED with appropriate error
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert "Conversation not found" in failed_call.kwargs["error_message"]
        assert failed_call.kwargs["error_code"] == 404

        # Verify message was still deleted from SQS
        sqs_client.delete_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_throttling_exception(self, mock_services, sample_message):
        """Test handling of AWS ThrottlingException (429)."""
        chat_service, job_repository, sqs_client = mock_services

        error_response = {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate limit exceeded"
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 400
            }
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify job was marked as FAILED with 429 error code
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert failed_call.kwargs["error_code"] == 429
        assert "ThrottlingException" in failed_call.kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_process_job_service_unavailable_exception(self, mock_services, sample_message):
        """Test handling of AWS ServiceUnavailableException (503)."""
        chat_service, job_repository, sqs_client = mock_services

        error_response = {
            "Error": {
                "Code": "ServiceUnavailableException",
                "Message": "Service temporarily unavailable"
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 503
            }
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify job was marked as FAILED with 503 error code
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert failed_call.kwargs["error_code"] == 503

    @pytest.mark.asyncio
    async def test_process_job_internal_server_exception(self, mock_services, sample_message):
        """Test handling of AWS InternalServerException (500)."""
        chat_service, job_repository, sqs_client = mock_services

        error_response = {
            "Error": {
                "Code": "InternalServerException",
                "Message": "Internal server error"
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 500
            }
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify job was marked as FAILED with 500 error code
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert failed_call.kwargs["error_code"] == 500

    @pytest.mark.asyncio
    async def test_process_job_generic_client_error(self, mock_services, sample_message):
        """Test handling of generic AWS ClientError."""
        chat_service, job_repository, sqs_client = mock_services

        error_response = {
            "Error": {
                "Code": "SomeOtherError",
                "Message": "Some error occurred"
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 502
            }
        }
        chat_service.execute_chat.side_effect = ClientError(
            error_response, "InvokeModel"
        )

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify error code from response metadata is used
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert failed_call.kwargs["error_code"] == 502

    @pytest.mark.asyncio
    async def test_process_job_generic_exception(self, mock_services, sample_message):
        """Test handling of generic exceptions."""
        chat_service, job_repository, sqs_client = mock_services

        chat_service.execute_chat.side_effect = Exception("Unexpected error")

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify job was marked as FAILED with 500 error code
        failed_call = [call for call in job_repository.update.call_args_list 
                      if call.kwargs.get("status") == job_models.JobStatus.FAILED][0]
        assert failed_call.kwargs["error_code"] == 500
        assert "Unexpected error" in failed_call.kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_process_job_deletes_message_on_exception(self, mock_services, sample_message):
        """Test that SQS message is always deleted even when an exception occurs."""
        chat_service, job_repository, sqs_client = mock_services

        chat_service.execute_chat.side_effect = Exception("Test error")

        await worker.process_job_message(
            sample_message, chat_service, job_repository, sqs_client
        )

        # Verify message was deleted despite the error
        sqs_client.delete_message.assert_called_once_with("test-receipt-handle")
