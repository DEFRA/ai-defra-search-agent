import uuid
from datetime import UTC, datetime

import pytest

from app.chat import job_models, job_repository


@pytest.fixture
def mock_client(mocker):
    """Create a mock AsyncIOMotorClient."""
    client = mocker.MagicMock()
    database = mocker.MagicMock()
    collection = mocker.AsyncMock()

    client.__getitem__.return_value = database
    database.chat_jobs = collection

    return client


@pytest.fixture
def job_repo(mock_client):
    """Create a MongoJobRepository with mocked client."""
    return job_repository.MongoJobRepository(mock_client, "test_db")


class TestMongoJobRepository:
    @pytest.mark.asyncio
    async def test_create_job(self, job_repo):
        """Test creating a new job."""
        job = job_models.ChatJob(
            question="What is AI?", model_id="anthropic.claude-3-haiku"
        )

        result = await job_repo.create(job)

        assert result == job
        job_repo.collection.insert_one.assert_called_once()
        inserted_doc = job_repo.collection.insert_one.call_args[0][0]
        assert inserted_doc["question"] == "What is AI?"
        assert inserted_doc["model_id"] == "anthropic.claude-3-haiku"
        assert inserted_doc["status"] == "queued"

    @pytest.mark.asyncio
    async def test_get_existing_job(self, job_repo):
        """Test retrieving an existing job."""
        job_id = uuid.uuid4()
        mock_doc = {
            "job_id": str(job_id),
            "conversation_id": None,
            "question": "Test question",
            "model_id": "test-model",
            "status": "completed",
            "result": {"conversation_id": "conv-123", "messages": []},
            "error_message": None,
            "error_code": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        job_repo.collection.find_one.return_value = mock_doc

        result = await job_repo.get(job_id)

        assert result is not None
        assert isinstance(result, job_models.ChatJob)
        assert result.job_id == job_id
        assert result.question == "Test question"
        assert result.status == job_models.JobStatus.COMPLETED
        job_repo.collection.find_one.assert_called_once_with({"job_id": str(job_id)})

    @pytest.mark.asyncio
    async def test_get_nonexistent_job_returns_none(self, job_repo):
        """Test retrieving a non-existent job returns None."""
        job_id = uuid.uuid4()
        job_repo.collection.find_one.return_value = None

        result = await job_repo.get(job_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_job_status(self, job_repo):
        """Test updating job status."""
        job_id = uuid.uuid4()
        mock_job = job_models.ChatJob(
            job_id=job_id,
            question="Test",
            model_id="test-model",
            status=job_models.JobStatus.PROCESSING,
        )

        # Mock the get call that happens after update
        job_repo.collection.find_one.return_value = mock_job.model_dump()

        await job_repo.update(job_id=job_id, status=job_models.JobStatus.PROCESSING)

        job_repo.collection.update_one.assert_called_once()
        update_call = job_repo.collection.update_one.call_args
        assert update_call[0][0] == {"job_id": str(job_id)}
        assert update_call[0][1]["$set"]["status"] == job_models.JobStatus.PROCESSING
        assert "updated_at" in update_call[0][1]["$set"]

    @pytest.mark.asyncio
    async def test_update_job_with_result(self, job_repo):
        """Test updating job with result."""
        job_id = uuid.uuid4()
        result_data = {
            "conversation_id": "conv-123",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        mock_job = job_models.ChatJob(
            job_id=job_id,
            question="Test",
            model_id="test-model",
            status=job_models.JobStatus.COMPLETED,
            result=result_data,
        )
        job_repo.collection.find_one.return_value = mock_job.model_dump()

        await job_repo.update(
            job_id=job_id, status=job_models.JobStatus.COMPLETED, result=result_data
        )

        update_call = job_repo.collection.update_one.call_args
        assert update_call[0][1]["$set"]["result"] == result_data
        assert update_call[0][1]["$set"]["status"] == job_models.JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_update_job_with_error(self, job_repo):
        """Test updating job with error details."""
        job_id = uuid.uuid4()

        mock_job = job_models.ChatJob(
            job_id=job_id,
            question="Test",
            model_id="test-model",
            status=job_models.JobStatus.FAILED,
            error_message="ThrottlingException: Rate limit exceeded",
            error_code=429,
        )
        job_repo.collection.find_one.return_value = mock_job.model_dump()

        await job_repo.update(
            job_id=job_id,
            status=job_models.JobStatus.FAILED,
            error_message="ThrottlingException: Rate limit exceeded",
            error_code=429,
        )

        update_call = job_repo.collection.update_one.call_args
        update_data = update_call[0][1]["$set"]
        assert (
            update_data["error_message"] == "ThrottlingException: Rate limit exceeded"
        )
        assert update_data["error_code"] == 429
        assert update_data["status"] == job_models.JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_update_only_includes_provided_fields(self, job_repo):
        """Test that update only includes fields that are provided."""
        job_id = uuid.uuid4()

        mock_job = job_models.ChatJob(
            job_id=job_id,
            question="Test",
            model_id="test-model",
            status=job_models.JobStatus.PROCESSING,
        )
        job_repo.collection.find_one.return_value = mock_job.model_dump()

        await job_repo.update(job_id=job_id, status=job_models.JobStatus.PROCESSING)

        update_call = job_repo.collection.update_one.call_args
        update_data = update_call[0][1]["$set"]

        # Should only have status and updated_at
        assert "status" in update_data
        assert "updated_at" in update_data
        assert "result" not in update_data
        assert "error_message" not in update_data
        assert "error_code" not in update_data

    @pytest.mark.asyncio
    async def test_create_job_with_conversation_id(self, job_repo):
        """Test creating a job with an existing conversation_id."""
        conversation_id = uuid.uuid4()
        job = job_models.ChatJob(
            conversation_id=conversation_id, question="Follow-up", model_id="test-model"
        )

        await job_repo.create(job)

        inserted_doc = job_repo.collection.insert_one.call_args[0][0]
        assert inserted_doc["conversation_id"] == str(conversation_id)

    @pytest.mark.asyncio
    async def test_job_timestamps_are_preserved(self, job_repo):
        """Test that job timestamps are preserved in storage."""
        created_time = datetime.now(UTC)
        job = job_models.ChatJob(
            question="Test",
            model_id="test-model",
            created_at=created_time,
            updated_at=created_time,
        )

        await job_repo.create(job)

        inserted_doc = job_repo.collection.insert_one.call_args[0][0]
        # Timestamps are serialized to ISO format strings in MongoDB
        assert "created_at" in inserted_doc
        assert "updated_at" in inserted_doc
        # Verify they were converted to strings
        assert isinstance(inserted_doc["created_at"], str)
        assert isinstance(inserted_doc["updated_at"], str)
