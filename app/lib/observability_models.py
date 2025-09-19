from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class NodeType(str, Enum):
    RETRIEVE = "retrieve"
    GRADE_DOCUMENTS = "grade_documents"
    GENERATE = "generate"
    GUARDRAILS_INPUT = "guardrails_input"
    GUARDRAILS_OUTPUT = "guardrails_output"


class AgentExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str
    query_hash: str
    status: ExecutionStatus

    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    total_duration_ms: int | None = None

    answer: str | None = None
    source_documents_count: int = 0
    source_documents: list[dict[str, Any]] = Field(default_factory=list)

    input_validation_passed: bool = True
    output_validation_passed: bool = True
    guardrails_violations: list[str] = Field(default_factory=list)
    security_events: list[str] = Field(default_factory=list)

    token_usage: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None

    error_message: str | None = None
    error_type: str | None = None
    stack_trace: str | None = None

    user_ip: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeExecution(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    node_name: str
    node_type: NodeType
    status: ExecutionStatus

    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: int | None = None

    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    input_size_bytes: int = 0
    output_size_bytes: int = 0

    retrieved_docs_count: int | None = None
    graded_docs_count: int | None = None
    generation_tokens: int | None = None

    error_message: str | None = None
    error_type: str | None = None
    retry_count: int = 0

    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None


class SystemMetrics(BaseModel):
    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    period_start: datetime
    period_end: datetime
    period_duration_minutes: int

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_executions: int = 0

    avg_execution_time_ms: float = 0.0
    p50_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0

    avg_retrieve_time_ms: float = 0.0
    avg_grade_time_ms: float = 0.0
    avg_generate_time_ms: float = 0.0

    avg_documents_retrieved: float = 0.0
    avg_answer_length: float = 0.0
    guardrails_block_rate: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_execution: float = 0.0

    top_error_types: list[dict[str, int]] = Field(default_factory=list)
    error_rate: float = 0.0

    security_events_count: int = 0
    injection_attempts_count: int = 0
    off_topic_queries_count: int = 0


class PerformanceAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    alert_type: str
    severity: str

    metric_name: str
    current_value: float
    threshold_value: float

    description: str
    affected_executions: list[str] = Field(default_factory=list)

    resolved: bool = False
    resolved_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
