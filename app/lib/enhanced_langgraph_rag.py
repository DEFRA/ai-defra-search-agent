import contextlib
from logging import getLogger

from fastapi import Depends
from langchain_core.callbacks import UsageMetadataCallbackHandler
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.common.tracing import ctx_trace_id
from app.lib.enhanced_security_monitor import ObservabilitySecurityMonitor
from app.lib.graph.graph import app
from app.lib.guardrails import GuardrailsManager
from app.lib.observability_service import (
    LangGraphObservabilityHandler,
    ObservabilityService,
)

logger = getLogger(__name__)


async def run_rag_llm_with_observability(query: str, db: AsyncDatabase = None):
    if db is None:
        from app.common.mongo import get_mongo_client

        client = await get_mongo_client()
        db = client.get_database("your_database_name")

    observability_service = ObservabilityService(db)
    security_monitor = ObservabilitySecurityMonitor(db)

    trace_id = None
    with contextlib.suppress(LookupError):
        trace_id = ctx_trace_id.get()

    execution = await observability_service.start_execution(query, trace_id)
    execution_id = execution.execution_id

    try:
        guardrails = GuardrailsManager()

        logger.info("Starting input validation", extra={"execution_id": execution_id})
        input_validation = guardrails.validate_input(query)

        await observability_service.update_validation_status(
            execution_id, input_valid=input_validation.is_valid
        )

        if not input_validation.is_valid:
            logger.warning(
                "Input validation failed",
                extra={"execution_id": execution_id, "reason": input_validation.reason},
            )

            if "injection" in input_validation.reason.lower():
                await security_monitor.log_injection_attempt_async(
                    query, input_validation.reason, execution_id
                )
            elif "off-topic" in input_validation.reason.lower():
                await security_monitor.log_off_topic_query_async(
                    query,
                    0.1,
                    execution_id,  # Low domain score
                )
            else:
                await security_monitor.log_validation_failure_async(
                    query, "input", input_validation.reason, execution_id
                )

            error_response = {
                "question": query,
                "answer": "I'm sorry, but I cannot process this request. Please ask a question related to Department for Environment, Food & Rural Affairs (Defra) and UK Government AI topics.",
                "source_documents": [],
                "usage": {},
                "validation_error": input_validation.reason,
                "validation_severity": input_validation.severity,
            }

            await observability_service.complete_execution(
                execution_id,
                answer=error_response["answer"],
                source_docs=[],
                token_usage={},
            )

            return error_response

        callback_handler = UsageMetadataCallbackHandler()
        observability_handler = LangGraphObservabilityHandler(
            observability_service, execution_id
        )

        config = {
            "callbacks": [callback_handler, observability_handler],
            "metadata": {"execution_id": execution_id},
        }

        logger.info(
            "Starting LangGraph execution", extra={"execution_id": execution_id}
        )
        response = app.invoke({"question": query}, config)

        logger.info("Starting output validation", extra={"execution_id": execution_id})
        output_validation = guardrails.validate_output(
            response["answer"], response["documents_for_context"], query
        )

        await observability_service.update_validation_status(
            execution_id, output_valid=output_validation.is_valid
        )

        if not output_validation.is_valid:
            logger.warning(
                "Output validation failed",
                extra={
                    "execution_id": execution_id,
                    "reason": output_validation.reason,
                },
            )

            await security_monitor.log_validation_failure_async(
                query, "output", output_validation.reason, execution_id
            )

            error_response = {
                "question": query,
                "answer": "I apologise, but I cannot provide a satisfactory answer based on the available documents. Please try rephrasing your question about Department for Environment, Food & Rural Affairs (Defra) and UK Government AI topics.",
                "source_documents": response["documents_for_context"],
                "usage": callback_handler.usage_metadata,
                "validation_error": output_validation.reason,
                "validation_severity": output_validation.severity,
            }

            await observability_service.complete_execution(
                execution_id,
                answer=error_response["answer"],
                source_docs=response["documents_for_context"],
                token_usage=callback_handler.usage_metadata,
            )

            return error_response

        logger.info(
            "RAG execution completed successfully",
            extra={
                "execution_id": execution_id,
                "documents_count": len(response["documents_for_context"]),
                "answer_length": len(response["answer"]),
            },
        )

        await security_monitor.log_successful_interaction_async(
            query,
            len(response["documents_for_context"]),
            len(response["answer"]),
            execution_id,
        )

        await observability_service.complete_execution(
            execution_id,
            answer=response["answer"],
            source_docs=response["documents_for_context"],
            token_usage=callback_handler.usage_metadata,
        )

        return {
            "question": response["question"],
            "answer": response["answer"],
            "source_documents": response["documents_for_context"],
            "usage": callback_handler.usage_metadata,
            "execution_id": execution_id,
        }

    except Exception as e:
        logger.error(
            "Error in RAG processing",
            extra={"execution_id": execution_id, "error": str(e)},
        )

        await security_monitor.log_system_error_async(query, str(e), execution_id)

        await observability_service.complete_execution(execution_id, error=e)

        return {
            "question": query,
            "answer": "I'm experiencing technical difficulties. Please try again later.",
            "source_documents": [],
            "usage": {},
            "error": str(e),
            "execution_id": execution_id,
        }


async def get_rag_service_with_observability(db: AsyncDatabase = Depends(get_db)):
    """Dependency injection for RAG service with observability."""

    async def rag_service(query: str):
        return await run_rag_llm_with_observability(query, db)

    return rag_service


class ObservabilityMetricsCollector:
    """Real-time metrics collector for live monitoring."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.metrics_collection = db.live_metrics

    async def collect_current_metrics(self) -> dict:
        """Collect current system metrics for live monitoring."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        one_hour_ago = now - timedelta(hours=1)

        recent_executions = await self.db.agent_executions.aggregate(
            [
                {"$match": {"timestamp": {"$gte": one_hour_ago}}},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "successful": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "failed": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                        "avg_duration": {"$avg": "$total_duration_ms"},
                        "avg_docs": {"$avg": "$source_documents_count"},
                    }
                },
            ]
        ).to_list(None)

        node_performance = await self.db.node_executions.aggregate(
            [
                {"$match": {"timestamp": {"$gte": one_hour_ago}}},
                {
                    "$group": {
                        "_id": "$node_type",
                        "avg_duration": {"$avg": "$duration_ms"},
                        "count": {"$sum": 1},
                    }
                },
            ]
        ).to_list(None)

        security_events = await self.db.security_events.aggregate(
            [
                {"$match": {"timestamp": {"$gte": one_hour_ago}}},
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            ]
        ).to_list(None)

        exec_stats = recent_executions[0] if recent_executions else {}

        metrics = {
            "timestamp": now.isoformat(),
            "period": "last_hour",
            "executions": {
                "total": exec_stats.get("total", 0),
                "successful": exec_stats.get("successful", 0),
                "failed": exec_stats.get("failed", 0),
                "success_rate": (
                    exec_stats.get("successful", 0) / exec_stats.get("total", 1) * 100
                ),
                "avg_duration_ms": round(exec_stats.get("avg_duration", 0) or 0, 2),
                "avg_documents_retrieved": round(exec_stats.get("avg_docs", 0) or 0, 2),
            },
            "nodes": {
                item["_id"]: {
                    "avg_duration_ms": round(item["avg_duration"] or 0, 2),
                    "count": item["count"],
                }
                for item in node_performance
            },
            "security": {item["_id"]: item["count"] for item in security_events},
        }

        await self.metrics_collection.insert_one(metrics)

        return metrics

    async def get_live_dashboard_data(self) -> dict:
        """Get data for a live monitoring dashboard."""
        current_metrics = await self.collect_current_metrics()

        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        twentyfour_hours_ago = now - timedelta(hours=24)

        hourly_trends = await self.db.agent_executions.aggregate(
            [
                {"$match": {"timestamp": {"$gte": twentyfour_hours_ago}}},
                {
                    "$group": {
                        "_id": {
                            "hour": {"$hour": "$timestamp"},
                            "date": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$timestamp",
                                }
                            },
                        },
                        "total": {"$sum": 1},
                        "successful": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "avg_duration": {"$avg": "$total_duration_ms"},
                    }
                },
                {"$sort": {"_id.date": 1, "_id.hour": 1}},
            ]
        ).to_list(None)

        return {
            "current": current_metrics,
            "trends": {"hourly": hourly_trends},
            "generated_at": now.isoformat(),
        }
