from logging import getLogger

from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.lib.graph.graph import app
from app.lib.guardrails import GuardrailsManager
from app.lib.security_monitoring import security_monitor

logger = getLogger(__name__)


def run_rag_llm(query: str):
    guardrails = GuardrailsManager()

    input_validation = guardrails.validate_input(query)
    if not input_validation.is_valid:
        logger.warning("Input validation failed: %s", input_validation.reason)
        return {
            "question": query,
            "answer": "I'm sorry, but I cannot process this request. Please ask a question related to Department for Environment, Food & Rural Affairs (Defra) and UK Government and UK Government AI topics.",
            "source_documents": [],
            "usage": {},
            "validation_error": input_validation.reason,
            "validation_severity": input_validation.severity,
        }

    callback_handler = UsageMetadataCallbackHandler()
    config = {"callbacks": [callback_handler]}

    try:
        response = app.invoke({"question": query}, config)

        output_validation = guardrails.validate_output(
            response["answer"], response["documents_for_context"], query
        )

        if not output_validation.is_valid:
            logger.warning("Output validation failed: %s", output_validation.reason)
            return {
                "question": query,
                "answer": "I apologise, but I cannot provide a satisfactory answer based on the available documents. Please try rephrasing your question about Department for Environment, Food & Rural Affairs (Defra) and UK Government AI topics.",
                "source_documents": response["documents_for_context"],
                "usage": callback_handler.usage_metadata,
                "validation_error": output_validation.reason,
                "validation_severity": output_validation.severity,
            }

        logger.info("Successful RAG interaction for query: %s", query[:100])
        security_monitor.log_successful_interaction(
            query, len(response["documents_for_context"]), len(response["answer"])
        )

        return {
            "question": response["question"],
            "answer": response["answer"],
            "source_documents": response["documents_for_context"],
            "usage": callback_handler.usage_metadata,
        }

    except Exception as e:
        logger.error("Error in RAG processing: %s", str(e))
        security_monitor.log_system_error(query, str(e))
        return {
            "question": query,
            "answer": "I'm experiencing technical difficulties. Please try again later.",
            "source_documents": [],
            "usage": {},
            "error": str(e),
        }
