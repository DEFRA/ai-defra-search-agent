from datetime import UTC, datetime, timedelta

from pymongo.asynchronous.database import AsyncDatabase

from app.lib.monitoring.security_monitoring import SecurityEvent, SecurityMonitor


class ObservabilitySecurityMonitor(SecurityMonitor):
    def __init__(self, db: AsyncDatabase):
        super().__init__()
        self.db = db
        self.security_events_collection = db.security_events
        self.executions_collection = db.agent_executions

    async def log_security_event_to_db(self, event: SecurityEvent) -> None:
        """Log security event to MongoDB for long-term analysis."""
        event_data = event.model_dump()
        await self.security_events_collection.insert_one(event_data)

        self.log_security_event(event)

    async def link_security_event_to_execution(
        self, execution_id: str, event_type: str, severity: str = "medium"
    ) -> None:
        """Link a security event to a specific execution for correlation."""
        await self.executions_collection.update_one(
            {"execution_id": execution_id},
            {"$push": {"security_events": f"{event_type}:{severity}"}},
        )

    async def log_injection_attempt_async(
        self, query: str, pattern_matched: str, execution_id: str | None = None
    ) -> None:
        """Log injection attempt with MongoDB storage."""
        event = SecurityEvent(
            event_type="injection_attempt",
            user_query=query,
            severity="high",
            details=f"Potential prompt injection detected. Pattern: {pattern_matched}",
            response_action="Request blocked by input validation",
            metadata={
                "pattern_matched": pattern_matched,
                "requires_investigation": True,
                "execution_id": execution_id,
            },
        )

        await self.log_security_event_to_db(event)

        if execution_id:
            await self.link_security_event_to_execution(
                execution_id, "injection_attempt", "high"
            )

    async def log_off_topic_query_async(
        self, query: str, domain_score: float, execution_id: str | None = None
    ) -> None:
        event = SecurityEvent(
            event_type="off_topic_query",
            user_query=query,
            severity="medium",
            details=f"Query appears to be off-topic (domain score: {domain_score})",
            response_action="Request rejected - outside domain scope",
            metadata={"domain_score": domain_score, "execution_id": execution_id},
        )

        await self.log_security_event_to_db(event)

        if execution_id:
            await self.link_security_event_to_execution(
                execution_id, "off_topic", "medium"
            )

    async def log_validation_failure_async(
        self,
        query: str,
        validation_type: str,
        reason: str,
        execution_id: str | None = None,
    ) -> None:
        event = SecurityEvent(
            event_type="validation_failure",
            user_query=query,
            severity="medium",
            details=f"{validation_type} validation failed: {reason}",
            response_action="Request processed with validation warnings",
            metadata={
                "validation_type": validation_type,
                "reason": reason,
                "execution_id": execution_id,
            },
        )

        await self.log_security_event_to_db(event)

        if execution_id:
            await self.link_security_event_to_execution(
                execution_id, "validation_failure", "medium"
            )

    async def log_successful_interaction_async(
        self,
        query: str,
        num_documents: int,
        response_length: int,
        execution_id: str | None = None,
    ) -> None:
        event = SecurityEvent(
            event_type="successful_interaction",
            user_query=query,
            severity="low",
            details=f"Successful RAG interaction: {num_documents} docs, {response_length} chars response",
            response_action="Request processed successfully",
            metadata={
                "documents_retrieved": num_documents,
                "response_length": response_length,
                "execution_id": execution_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        await self.log_security_event_to_db(event)

        self.log_successful_interaction(query, num_documents, response_length)

    async def log_system_error_async(
        self, query: str, error_message: str, execution_id: str | None = None
    ) -> None:
        """Log system error with execution correlation."""
        event = SecurityEvent(
            event_type="system_error",
            user_query=query,
            severity="high",
            details=f"System error during processing: {error_message}",
            response_action="Error logged for investigation",
            metadata={
                "error_message": error_message,
                "execution_id": execution_id,
                "requires_investigation": True,
            },
        )

        await self.log_security_event_to_db(event)

        self.log_system_error(query, error_message)

    async def get_security_metrics(self, hours_back: int = 24) -> dict:
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)

        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_time}}},
            {
                "$group": {
                    "_id": "$event_type",
                    "count": {"$sum": 1},
                    "high_severity": {
                        "$sum": {"$cond": [{"$eq": ["$severity", "high"]}, 1, 0]}
                    },
                    "medium_severity": {
                        "$sum": {"$cond": [{"$eq": ["$severity", "medium"]}, 1, 0]}
                    },
                    "low_severity": {
                        "$sum": {"$cond": [{"$eq": ["$severity", "low"]}, 1, 0]}
                    },
                }
            },
        ]

        cursor = await self.security_events_collection.aggregate(pipeline)
        results = await cursor.to_list(None)

        metrics = {
            "time_period_hours": hours_back,
            "cutoff_time": cutoff_time.isoformat(),
            "events_by_type": {},
            "total_events": 0,
            "total_high_severity": 0,
            "total_medium_severity": 0,
            "total_low_severity": 0,
        }

        for result in results:
            event_type = result["_id"]
            metrics["events_by_type"][event_type] = {
                "total": result["count"],
                "high_severity": result["high_severity"],
                "medium_severity": result["medium_severity"],
                "low_severity": result["low_severity"],
            }

            metrics["total_events"] += result["count"]
            metrics["total_high_severity"] += result["high_severity"]
            metrics["total_medium_severity"] += result["medium_severity"]
            metrics["total_low_severity"] += result["low_severity"]

        return metrics

    async def get_recent_security_events(self, limit: int = 50) -> list[dict]:
        events = await self.security_events_collection.find(
            {}, sort=[("timestamp", -1)], limit=limit
        ).to_list(None)

        for event in events:
            if "_id" in event:
                event["_id"] = str(event["_id"])

        return events

    async def get_security_trends(self, days_back: int = 7) -> dict:
        cutoff_time = datetime.now(UTC) - timedelta(days=days_back)

        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_time}}},
            {
                "$group": {
                    "_id": {
                        "date": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp",
                            }
                        },
                        "event_type": "$event_type",
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.date": 1}},
        ]

        cursor = await self.security_events_collection.aggregate(pipeline)
        results = await cursor.to_list(None)

        trends = {}
        for result in results:
            date = result["_id"]["date"]
            event_type = result["_id"]["event_type"]

            if date not in trends:
                trends[date] = {}

            trends[date][event_type] = result["count"]

        return {
            "period_days": days_back,
            "cutoff_time": cutoff_time.isoformat(),
            "daily_trends": trends,
        }
