import hashlib
import json
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from functools import wraps
from logging import getLogger
from typing import Any

from fastapi import Depends
from langchain_core.callbacks import BaseCallbackHandler
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.common.tracing import ctx_trace_id
from app.lib.observability.observability_models import (
    AgentExecution,
    ExecutionStatus,
    NodeExecution,
    NodeType,
)

logger = getLogger(__name__)


class ObservabilityService:
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.executions_collection = db.agent_executions
        self.nodes_collection = db.node_executions
        self.metrics_collection = db.system_metrics
        self.alerts_collection = db.performance_alerts

    async def start_execution(
        self, query: str, trace_id: str | None = None
    ) -> AgentExecution:
        execution = AgentExecution(
            query=query,
            query_hash=self._hash_query(query),
            status=ExecutionStatus.STARTED,
            trace_id=trace_id,
        )

        await self.executions_collection.insert_one(execution.model_dump())

        logger.info(
            "Started execution tracking",
            extra={
                "execution_id": execution.execution_id,
                "query_preview": query[:100],
                "observability": True,
            },
        )

        return execution

    async def complete_execution(
        self,
        execution_id: str,
        answer: str | None = None,
        source_docs: list[dict[str, Any]] | None = None,
        token_usage: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        end_time = datetime.now(UTC)

        execution_doc = await self.executions_collection.find_one(
            {"execution_id": execution_id}
        )

        if not execution_doc:
            logger.warning(
                "Execution not found for completion",
                extra={"execution_id": execution_id, "observability": True},
            )
            return

        start_time = execution_doc["start_time"]

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)

        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        update_data = {
            "end_time": end_time,
            "total_duration_ms": duration_ms,
        }

        if error:
            update_data.update(
                {
                    "status": ExecutionStatus.FAILED.value,
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                    "stack_trace": traceback.format_exc(),
                }
            )
        else:
            serialized_docs = []
            if source_docs:
                for doc in source_docs:
                    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                        serialized_docs.append(
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                                "id": getattr(doc, "id", None),
                            }
                        )
                    else:
                        serialized_docs.append(doc)
            update_data.update(
                {
                    "status": ExecutionStatus.COMPLETED.value,
                    "answer": answer,
                    "source_documents_count": len(source_docs or []),
                    "source_documents": serialized_docs,
                    "token_usage": token_usage or {},
                }
            )

        await self.executions_collection.update_one(
            {"execution_id": execution_id}, {"$set": update_data}
        )

        logger.info(
            "Completed execution tracking",
            extra={
                "execution_id": execution_id,
                "duration_ms": duration_ms,
                "status": "failed" if error else "completed",
                "observability": True,
            },
        )

    def _safe_json_serialize(self, obj):
        """Safely serialize an object to JSON, handling non-serializable types."""
        if obj is None:
            return {}

        # Handle non-dict objects first to reduce complexity
        if not isinstance(obj, dict):
            # For non-dict objects, convert to string
            return {"value": str(obj)}

        # Create a sanitized copy to avoid modifying the original
        sanitized = {}
        for k, v in obj.items():
            try:
                # Try simple serialization first
                json.dumps({k: v})
                sanitized[k] = v
            except (TypeError, OverflowError):
                # If it fails, handle various object types
                sanitized[k] = self._convert_unserializable_value(v)

        return sanitized

    def _convert_unserializable_value(self, v):
        """Helper method to convert an unserializable value to a serializable format."""
        # Handle Pydantic models
        if hasattr(v, "model_dump"):
            try:
                return v.model_dump()
            except Exception:
                return str(v)

        # Handle objects with dict() method
        if hasattr(v, "dict") and callable(v.dict):
            try:
                return v.dict()
            except Exception:
                return str(v)

        # Handle objects with __dict__
        if hasattr(v, "__dict__"):
            try:
                return str(v)
            except Exception:
                return f"<non-serializable {type(v).__name__}>"

        # Default case
        return f"<non-serializable {type(v).__name__}>"

    async def track_node_execution(
        self,
        execution_id: str,
        node_name: str,
        node_type: NodeType,
        status: ExecutionStatus,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        error: Exception | None = None,
        **node_specific_metrics,
    ) -> None:
        # Sanitize input and output data for serialization
        safe_input_data = self._safe_json_serialize(input_data)
        safe_output_data = self._safe_json_serialize(output_data)

        # Calculate sizes using the sanitized versions
        try:
            input_size_bytes = len(json.dumps(safe_input_data).encode())
            output_size_bytes = len(json.dumps(safe_output_data).encode())
        except (TypeError, OverflowError):
            # Fallback if serialization still fails
            input_size_bytes = len(str(input_data).encode()) if input_data else 0
            output_size_bytes = len(str(output_data).encode()) if output_data else 0

        node_execution = NodeExecution(
            execution_id=execution_id,
            node_name=node_name,
            node_type=node_type,
            status=status,
            input_data=safe_input_data,
            output_data=safe_output_data,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
            duration_ms=duration_ms,
            error_message=str(error) if error else None,
            error_type=type(error).__name__ if error else None,
            **node_specific_metrics,
        )

        await self.nodes_collection.insert_one(node_execution.model_dump())

        logger.debug(
            "Tracked node execution",
            extra={
                "execution_id": execution_id,
                "node_name": node_name,
                "node_type": node_type.value,
                "status": status.value,
                "duration_ms": duration_ms,
                "observability": True,
            },
        )

    async def update_validation_status(
        self,
        execution_id: str,
        input_valid: bool | None = None,
        output_valid: bool | None = None,
        violations: list[str] | None = None,
    ) -> None:
        update_data = {}

        if input_valid is not None:
            update_data["input_validation_passed"] = input_valid

        if output_valid is not None:
            update_data["output_validation_passed"] = output_valid

        if violations:
            update_data["guardrails_violations"] = violations

        if update_data:
            await self.executions_collection.update_one(
                {"execution_id": execution_id}, {"$set": update_data}
            )

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]


class LangGraphObservabilityHandler(BaseCallbackHandler):
    def __init__(self, observability_service: ObservabilityService, execution_id: str):
        self.observability_service = observability_service
        self.execution_id = execution_id
        self.node_start_times = {}

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **_kwargs
    ) -> None:
        node_name = serialized.get("name", "unknown_node")
        self.node_start_times[node_name] = time.time()

        node_type_mapping = {
            "retrieve": NodeType.RETRIEVE,
            "grade_documents": NodeType.GRADE_DOCUMENTS,
            "generate": NodeType.GENERATE,
        }

        node_type = node_type_mapping.get(node_name, NodeType.RETRIEVE)

        import asyncio

        asyncio.create_task(
            self.observability_service.track_node_execution(
                execution_id=self.execution_id,
                node_name=node_name,
                node_type=node_type,
                status=ExecutionStatus.STARTED,
                input_data=inputs,
            )
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs) -> None:
        # Extract node name from serialized data if available, otherwise use kwargs
        serialized = kwargs.get("serialized", {})
        node_name = serialized.get("name") if serialized else None

        # Fallback to checking run_id which might contain node name
        if not node_name and "run_id" in kwargs:
            parts = kwargs["run_id"].split("-")
            if len(parts) > 1:
                node_name = parts[0]

        # Final fallback
        if not node_name:
            node_name = "unknown_node"

        start_time = self.node_start_times.get(node_name)
        duration_ms = int((time.time() - start_time) * 1000) if start_time else None

        node_type_mapping = {
            "retrieve": NodeType.RETRIEVE,
            "grade_documents": NodeType.GRADE_DOCUMENTS,
            "generate": NodeType.GENERATE,
        }

        node_type = node_type_mapping.get(node_name, NodeType.RETRIEVE)

        import asyncio

        asyncio.create_task(
            self.observability_service.track_node_execution(
                execution_id=self.execution_id,
                node_name=node_name,
                node_type=node_type,
                status=ExecutionStatus.COMPLETED,
                output_data=outputs,
                duration_ms=duration_ms,
            )
        )

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        # Extract node name from serialized data if available, otherwise use kwargs
        serialized = kwargs.get("serialized", {})
        node_name = serialized.get("name") if serialized else None

        # Fallback to checking run_id which might contain node name
        if not node_name and "run_id" in kwargs:
            parts = kwargs["run_id"].split("-")
            if len(parts) > 1:
                node_name = parts[0]

        # Final fallback
        if not node_name:
            node_name = "unknown_node"

        start_time = self.node_start_times.get(node_name)
        duration_ms = int((time.time() - start_time) * 1000) if start_time else None

        node_type_mapping = {
            "retrieve": NodeType.RETRIEVE,
            "grade_documents": NodeType.GRADE_DOCUMENTS,
            "generate": NodeType.GENERATE,
        }

        node_type = node_type_mapping.get(node_name, NodeType.RETRIEVE)

        import asyncio

        asyncio.create_task(
            self.observability_service.track_node_execution(
                execution_id=self.execution_id,
                node_name=node_name,
                node_type=node_type,
                status=ExecutionStatus.FAILED,
                duration_ms=duration_ms,
                error=error,
            )
        )


def track_node_performance(node_name: str, node_type: NodeType):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    "Node execution completed",
                    extra={
                        "node_name": node_name,
                        "node_type": node_type.value,
                        "duration_ms": duration_ms,
                        "status": "success",
                        "performance_tracking": True,
                    },
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                logger.error(
                    "Node execution failed",
                    extra={
                        "node_name": node_name,
                        "node_type": node_type.value,
                        "duration_ms": duration_ms,
                        "error": str(e),
                        "status": "failed",
                        "performance_tracking": True,
                    },
                )

                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    "Node execution completed",
                    extra={
                        "node_name": node_name,
                        "node_type": node_type.value,
                        "duration_ms": duration_ms,
                        "status": "success",
                        "performance_tracking": True,
                    },
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                logger.error(
                    "Node execution failed",
                    extra={
                        "node_name": node_name,
                        "node_type": node_type.value,
                        "duration_ms": duration_ms,
                        "error": str(e),
                        "status": "failed",
                        "performance_tracking": True,
                    },
                )

                raise

        if hasattr(func, "__code__") and "await" in func.__code__.co_names:
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def execution_context(observability_service: ObservabilityService, query: str):
    execution = None
    trace_id = None

    try:
        with suppress(LookupError):
            trace_id = ctx_trace_id.get()

        import asyncio

        execution = asyncio.run(observability_service.start_execution(query, trace_id))

        logger.info(
            "Agent execution started",
            extra={
                "execution_id": execution.execution_id,
                "trace_id": trace_id,
                "query_preview": query[:100],
                "agent_execution": True,
            },
        )

        yield execution.execution_id

    except Exception as e:
        if execution:
            asyncio.run(
                observability_service.complete_execution(
                    execution.execution_id, error=e
                )
            )
        raise

    else:
        pass


async def get_observability_service(
    db: AsyncDatabase = Depends(get_db),
) -> ObservabilityService:
    return ObservabilityService(db)
