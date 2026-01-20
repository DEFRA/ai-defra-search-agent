import enum
import uuid
from datetime import UTC, datetime
from typing import Any

import pydantic


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatJob(pydantic.BaseModel):
    job_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID | None = None
    question: str
    model_id: str
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: int | None = None
    created_at: datetime = pydantic.Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = pydantic.Field(default_factory=lambda: datetime.now(UTC))

    class Config:
        use_enum_values = True
