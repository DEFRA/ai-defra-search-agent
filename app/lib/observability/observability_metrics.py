from pymongo.asynchronous.database import AsyncDatabase


class ObservabilityMetricsCollector:
    """
    Real-time metrics collector for live monitoring and dashboarding.
    """

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.metrics_collection = db.live_metrics

    async def collect_current_metrics(self) -> dict:
        """
        Collect current system metrics for live monitoring.
        Returns a dictionary of metrics for the last hour.
        """
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
                )
                if exec_stats.get("total", 0)
                else 0.0,
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
        """
        Get data for a live monitoring dashboard, including current and hourly trend metrics.
        """
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
