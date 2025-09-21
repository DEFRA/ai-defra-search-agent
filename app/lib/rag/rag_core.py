import contextlib
from datetime import UTC, datetime
from logging import getLogger
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from pymongo.asynchronous.database import AsyncDatabase

from app.common.tracing import ctx_trace_id
from app.lib.conversation_history.service import ConversationHistoryService
from app.lib.graph.graph import app
from app.lib.guardrails.guardrails import GuardrailsManager
from app.lib.monitoring.enhanced_security_monitor import ObservabilitySecurityMonitor
from app.lib.observability.observability_service import (
    LangGraphObservabilityHandler,
    ObservabilityService,
)

logger = getLogger(__name__)


def build_error_response(
    question: str,
    answer: str,
    validation_error: str | None = None,
    validation_severity: str | None = None,
    source_documents: list | None = None,
    usage: dict | None = None,
    error: str | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "source_documents": source_documents or [],
        "usage": usage or {},
        "validation_error": validation_error,
        "validation_severity": validation_severity,
        "error": error,
        "execution_id": execution_id,
    }


def get_trace_id() -> str | None:
    with contextlib.suppress(LookupError):
        return ctx_trace_id.get()
    return None


async def validate_input(
    query: str,
    guardrails: GuardrailsManager,
    observability_service: ObservabilityService,
    security_monitor: ObservabilitySecurityMonitor,
    execution_id: str,
):
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
        reason = input_validation.reason.lower()
        if "injection" in reason:
            await security_monitor.log_injection_attempt_async(
                query, input_validation.reason, execution_id
            )
        elif "off-topic" in reason:
            await security_monitor.log_off_topic_query_async(query, 0.1, execution_id)
        else:
            await security_monitor.log_validation_failure_async(
                query, "input", input_validation.reason, execution_id
            )
        return build_error_response(
            question=query,
            answer="I'm sorry, but I cannot process this request. Please ask a question related to Department for Environment, Food & Rural Affairs (Defra) and UK Government AI topics.",
            validation_error=input_validation.reason,
            validation_severity=input_validation.severity,
            execution_id=execution_id,
        )
    return None


async def validate_output(
    query: str,
    response: dict,
    guardrails: GuardrailsManager,
    observability_service: ObservabilityService,
    security_monitor: ObservabilitySecurityMonitor,
    callback_handler: UsageMetadataCallbackHandler,
    execution_id: str,
):
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
            extra={"execution_id": execution_id, "reason": output_validation.reason},
        )
        await security_monitor.log_validation_failure_async(
            query, "output", output_validation.reason, execution_id
        )
        return build_error_response(
            question=query,
            answer="I apologise, but I cannot provide a satisfactory answer based on the available documents. Please try rephrasing your question about Department for Environment, Food & Rural Affairs (Defra) and UK Government AI topics.",
            validation_error=output_validation.reason,
            validation_severity=output_validation.severity,
            source_documents=response["documents_for_context"],
            usage=callback_handler.usage_metadata,
            execution_id=execution_id,
        )
    return None


async def run_rag_llm_with_observability(  # noqa: C901
    query: str, db: AsyncDatabase = None, conversation_id: str = None
):
    """
    Run RAG LLM with observability, validation, and security monitoring.
    Handles input/output validation, observability, and error handling.
    """
    if db is None:
        from app.common.mongo import get_mongo_client

        client = await get_mongo_client()
        db = client.get_database("your_database_name")

    observability_service = ObservabilityService(db)
    security_monitor = ObservabilitySecurityMonitor(db)
    trace_id = get_trace_id()
    execution = await observability_service.start_execution(query, trace_id)
    execution_id = execution.execution_id
    guardrails = GuardrailsManager()

    # Conversation history logic
    conversation_service = ConversationHistoryService(db)
    if not conversation_id:
        conversation_id = str(uuid4())
        await conversation_service.create_conversation(conversation_id)
    history_doc = await conversation_service.get_history(conversation_id)
    conversation_history = (
        history_doc["messages"] if history_doc and "messages" in history_doc else []
    )

    try:
        # Input validation
        input_error = await validate_input(
            query, guardrails, observability_service, security_monitor, execution_id
        )
        if input_error:
            await observability_service.complete_execution(
                execution_id,
                answer=input_error["answer"],
                source_docs=[],
                token_usage={},
            )
            return input_error

        # RAG execution
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
        state = {"question": query, "conversation_history": conversation_history}
        response = app.invoke(state, config)

        # Update conversation history with new message
        new_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if "answer" in response:
            new_message["answer"] = response["answer"]

        if "documents_for_context" in response:
            docs = response["documents_for_context"]
            sources = []
            if docs and isinstance(docs, list):
                seen = set()
                for doc in docs:
                    title = None
                    url = None
                    if hasattr(doc, "metadata"):
                        title = doc.metadata.get("title")
                        url = doc.metadata.get("url")
                    if not title and hasattr(doc, "title"):
                        title = doc.title
                    if not url and hasattr(doc, "url"):
                        url = doc.url
                    key = (title, url)
                    if title and key not in seen:
                        sources.append({"title": title, "url": url})
                        seen.add(key)
                new_message["sources"] = sources
            else:
                new_message["sources"] = docs

        await conversation_service.add_message(conversation_id, new_message)
        updated_history_doc = await conversation_service.get_history(conversation_id)
        updated_conversation_history = (
            updated_history_doc["messages"]
            if updated_history_doc and "messages" in updated_history_doc
            else []
        )

        # Output validation
        output_error = await validate_output(
            query,
            response,
            guardrails,
            observability_service,
            security_monitor,
            callback_handler,
            execution_id,
        )
        if output_error:
            await observability_service.complete_execution(
                execution_id,
                answer=output_error["answer"],
                source_docs=response["documents_for_context"],
                token_usage=callback_handler.usage_metadata,
            )
            return output_error

        # Success
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
            "conversation_id": conversation_id,
            "conversation_history": updated_conversation_history,
        }

    except Exception as e:
        logger.error(
            "Error in RAG processing",
            extra={"execution_id": execution_id, "error": str(e)},
        )
        await security_monitor.log_system_error_async(query, str(e), execution_id)
        await observability_service.complete_execution(execution_id, error=e)
        return build_error_response(
            question=query,
            answer="I'm experiencing technical difficulties. Please try again later.",
            error=str(e),
            execution_id=execution_id,
        )
