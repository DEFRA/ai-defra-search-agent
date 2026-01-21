import uuid
from datetime import UTC, datetime

from app.chat import job_models


class TestJobStatus:
    def test_job_status_enum_values(self):
        """Test that JobStatus enum has expected values."""
        assert job_models.JobStatus.QUEUED == "queued"
        assert job_models.JobStatus.PROCESSING == "processing"
        assert job_models.JobStatus.COMPLETED == "completed"
        assert job_models.JobStatus.FAILED == "failed"

    def test_job_status_is_string_enum(self):
        """Test that JobStatus values are strings."""
        assert isinstance(job_models.JobStatus.QUEUED.value, str)
        assert isinstance(job_models.JobStatus.PROCESSING.value, str)


class TestChatJob:
    def test_chat_job_creation_with_defaults(self):
        """Test creating a ChatJob with only required fields."""
        job = job_models.ChatJob(
            question="What is AI?", model_id="anthropic.claude-3-haiku"
        )

        assert isinstance(job.job_id, uuid.UUID)
        assert job.conversation_id is None
        assert job.question == "What is AI?"
        assert job.model_id == "anthropic.claude-3-haiku"
        assert job.status == job_models.JobStatus.QUEUED
        assert job.result is None
        assert job.error_message is None
        assert job.error_code is None
        assert isinstance(job.created_at, datetime)
        assert isinstance(job.updated_at, datetime)

    def test_chat_job_creation_with_conversation_id(self):
        """Test creating a ChatJob with an existing conversation_id."""
        conversation_id = uuid.uuid4()
        job = job_models.ChatJob(
            conversation_id=conversation_id,
            question="Follow-up question?",
            model_id="anthropic.claude-3-haiku",
        )

        assert job.conversation_id == conversation_id

    def test_chat_job_with_custom_status(self):
        """Test creating a ChatJob with a custom status."""
        job = job_models.ChatJob(
            question="Test question",
            model_id="test-model",
            status=job_models.JobStatus.PROCESSING,
        )

        assert job.status == job_models.JobStatus.PROCESSING

    def test_chat_job_with_result(self):
        """Test creating a ChatJob with a result."""
        result = {
            "conversation_id": "test-conv-id",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        job = job_models.ChatJob(
            question="Hello",
            model_id="test-model",
            status=job_models.JobStatus.COMPLETED,
            result=result,
        )

        assert job.result == result
        assert job.status == job_models.JobStatus.COMPLETED

    def test_chat_job_with_error(self):
        """Test creating a ChatJob with error details."""
        job = job_models.ChatJob(
            question="Test question",
            model_id="test-model",
            status=job_models.JobStatus.FAILED,
            error_message="ThrottlingException: Rate limit exceeded",
            error_code=429,
        )

        assert job.status == job_models.JobStatus.FAILED
        assert job.error_message == "ThrottlingException: Rate limit exceeded"
        assert job.error_code == 429

    def test_chat_job_model_dump(self):
        """Test serializing ChatJob to dict."""
        job = job_models.ChatJob(question="Test", model_id="test-model")

        dumped = job.model_dump()

        assert "job_id" in dumped
        assert "question" in dumped
        assert "model_id" in dumped
        assert "status" in dumped
        assert "created_at" in dumped
        assert "updated_at" in dumped

    def test_chat_job_timestamps_auto_set(self):
        """Test that timestamps are automatically set to UTC."""
        before = datetime.now(UTC)
        job = job_models.ChatJob(question="Test", model_id="test-model")
        after = datetime.now(UTC)

        assert before <= job.created_at <= after
        assert before <= job.updated_at <= after
        assert job.created_at.tzinfo == UTC
        assert job.updated_at.tzinfo == UTC

    def test_chat_job_enum_values_serialized_as_strings(self):
        """Test that JobStatus enum values are serialized as strings."""
        job = job_models.ChatJob(
            question="Test",
            model_id="test-model",
            status=job_models.JobStatus.PROCESSING,
        )

        dumped = job.model_dump()
        assert dumped["status"] == "processing"
        assert isinstance(dumped["status"], str)
