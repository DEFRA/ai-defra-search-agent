import json
from datetime import datetime
from enum import Enum
from logging import getLogger

from pydantic import BaseModel, Field

logger = getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events to monitor."""

    INJECTION_ATTEMPT = "injection_attempt"
    OFF_TOPIC_QUERY = "off_topic_query"
    VALIDATION_FAILURE = "validation_failure"
    KNOWLEDGE_LEAKAGE = "knowledge_leakage"
    SUCCESSFUL_INTERACTION = "successful_interaction"
    SYSTEM_ERROR = "system_error"


class SecurityEvent(BaseModel):
    """Security event model for tracking incidents."""

    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: SecurityEventType
    user_query: str = Field(description="User query (truncated if sensitive)")
    severity: str = Field(description="Severity level: low, medium, high")
    details: str = Field(description="Event details and context")
    response_action: str = Field(description="Action taken in response")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SecurityMonitor:
    """Monitor and log security-related events."""

    def __init__(self):
        self.logger = getLogger(f"{__name__}.SecurityMonitor")

    def log_security_event(self, event: SecurityEvent) -> None:
        """Log a security event with appropriate level."""

        sanitized_query = self._sanitize_for_logging(event.user_query)

        log_data = {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "query_preview": sanitized_query[:200] + "..."
            if len(sanitized_query) > 200
            else sanitized_query,
            "severity": event.severity,
            "details": event.details,
            "response_action": event.response_action,
            "metadata": event.metadata,
        }

        if event.severity == "high":
            self.logger.error("SECURITY EVENT: %s", json.dumps(log_data))
        elif event.severity == "medium":
            self.logger.warning("SECURITY EVENT: %s", json.dumps(log_data))
        else:
            self.logger.info("SECURITY EVENT: %s", json.dumps(log_data))

    def log_injection_attempt(
        self, query: str, details: str, severity: str = "high"
    ) -> None:
        """Log a potential prompt injection attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.INJECTION_ATTEMPT,
            user_query=query,
            severity=severity,
            details=details,
            response_action="Request blocked by input validation",
            metadata={"pattern_matched": True},
        )
        self.log_security_event(event)

    def log_off_topic_query(
        self, query: str, details: str, severity: str = "medium"
    ) -> None:
        """Log an off-topic query attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.OFF_TOPIC_QUERY,
            user_query=query,
            severity=severity,
            details=details,
            response_action="Query rejected as off-topic",
            metadata={"topic_classification": "outside_domain"},
        )
        self.log_security_event(event)

    def log_validation_failure(
        self, query: str, validation_type: str, details: str, severity: str = "medium"
    ) -> None:
        """Log a validation failure."""
        event = SecurityEvent(
            event_type=SecurityEventType.VALIDATION_FAILURE,
            user_query=query,
            severity=severity,
            details=f"{validation_type}: {details}",
            response_action="Request rejected by validation",
            metadata={"validation_type": validation_type},
        )
        self.log_security_event(event)

    def log_knowledge_leakage(
        self, query: str, response: str, details: str, severity: str = "medium"
    ) -> None:
        """Log potential knowledge leakage."""
        event = SecurityEvent(
            event_type=SecurityEventType.KNOWLEDGE_LEAKAGE,
            user_query=query,
            severity=severity,
            details=details,
            response_action="Response filtered or blocked",
            metadata={
                "response_preview": response[:100] + "..."
                if len(response) > 100
                else response,
                "leakage_detected": True,
            },
        )
        self.log_security_event(event)

    def log_successful_interaction(
        self, query: str, num_sources: int, response_length: int
    ) -> None:
        """Log a successful, validated interaction."""
        event = SecurityEvent(
            event_type=SecurityEventType.SUCCESSFUL_INTERACTION,
            user_query=query,
            severity="low",
            details="Query processed successfully with validation",
            response_action="Response provided",
            metadata={
                "source_documents_used": num_sources,
                "response_length": response_length,
                "all_validations_passed": True,
            },
        )
        self.log_security_event(event)

    def log_system_error(
        self, query: str, error: str, severity: str = "medium"
    ) -> None:
        """Log system errors that might indicate security issues."""
        event = SecurityEvent(
            event_type=SecurityEventType.SYSTEM_ERROR,
            user_query=query,
            severity=severity,
            details=f"System error: {error}",
            response_action="Generic error response provided",
            metadata={
                "error_type": type(error).__name__
                if isinstance(error, Exception)
                else "unknown"
            },
        )
        self.log_security_event(event)

    def _sanitize_for_logging(self, text: str) -> str:
        """Sanitize text for safe logging (remove potential sensitive patterns)."""
        sensitive_patterns = [
            r"password[:=]\s*\w+",
            r"api[_-]?key[:=]\s*\w+",
            r"token[:=]\s*\w+",
            r"secret[:=]\s*\w+",
        ]

        sanitized = text
        for pattern in sensitive_patterns:
            import re

            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

        return sanitized


security_monitor = SecurityMonitor()
