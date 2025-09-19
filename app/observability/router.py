from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from app.common.mongo import get_db
from app.lib.enhanced_security_monitor import ObservabilitySecurityMonitor

observability_router = APIRouter(prefix="/observability", tags=["observability"])

router = observability_router


@observability_router.get("/health")
async def health_check():
    """Basic health check for observability system."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "observability",
    }


@observability_router.get("/executions/recent")
async def get_recent_executions(
    limit: int = Query(default=50, description="Number of recent executions to return"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get recent agent executions with basic metrics."""
    try:
        executions = await db.agent_executions.find(
            {},
            {
                "execution_id": 1,
                "timestamp": 1,
                "status": 1,
                "total_duration_ms": 1,
                "query_hash": 1,
                "source_documents_count": 1,
                "input_validation_passed": 1,
                "output_validation_passed": 1,
                "error_message": 1,
            },
            sort=[("timestamp", -1)],
            limit=limit,
        ).to_list(None)

        for execution in executions:
            if "_id" in execution:
                execution["_id"] = str(execution["_id"])

        return {
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch executions: {str(e)}"
        ) from e


@observability_router.get("/executions/{execution_id}")
async def get_execution_details(execution_id: str, db: AsyncDatabase = Depends(get_db)):
    """Get detailed information about a specific execution."""
    try:
        execution = await db.agent_executions.find_one({"execution_id": execution_id})

        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        nodes = await db.node_executions.find(
            {"execution_id": execution_id}, sort=[("timestamp", 1)]
        ).to_list(None)

        execution["_id"] = str(execution["_id"])
        for node in nodes:
            node["_id"] = str(node["_id"])

        return {
            "execution": execution,
            "nodes": nodes,
            "node_count": len(nodes),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch execution details: {str(e)}"
        ) from e


@observability_router.get("/metrics/performance")
async def get_performance_metrics(
    hours: int = Query(default=24, description="Hours of data to analyze"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get performance metrics for the specified time period."""
    try:
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_time}}},
            {
                "$group": {
                    "_id": None,
                    "total_executions": {"$sum": 1},
                    "successful_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "failed_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                    },
                    "avg_duration_ms": {"$avg": "$total_duration_ms"},
                    "min_duration_ms": {"$min": "$total_duration_ms"},
                    "max_duration_ms": {"$max": "$total_duration_ms"},
                    "avg_documents_retrieved": {"$avg": "$source_documents_count"},
                    "validation_failures": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$or": [
                                        {"$eq": ["$input_validation_passed", False]},
                                        {"$eq": ["$output_validation_passed", False]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]

        cursor = await db.agent_executions.aggregate(pipeline)
        result = await cursor.to_list(None)

        if not result:
            return {
                "period_hours": hours,
                "cutoff_time": cutoff_time.isoformat(),
                "metrics": {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "success_rate": 0.0,
                    "avg_duration_ms": 0.0,
                },
            }

        metrics = result[0]
        total = metrics.get("total_executions", 0)
        successful = metrics.get("successful_executions", 0)

        return {
            "period_hours": hours,
            "cutoff_time": cutoff_time.isoformat(),
            "metrics": {
                "total_executions": total,
                "successful_executions": successful,
                "failed_executions": metrics.get("failed_executions", 0),
                "success_rate": (successful / total * 100) if total > 0 else 0.0,
                "avg_duration_ms": round(metrics.get("avg_duration_ms", 0) or 0, 2),
                "min_duration_ms": metrics.get("min_duration_ms", 0),
                "max_duration_ms": metrics.get("max_duration_ms", 0),
                "avg_documents_retrieved": round(
                    metrics.get("avg_documents_retrieved", 0) or 0, 2
                ),
                "validation_failure_rate": (
                    metrics.get("validation_failures", 0) / total * 100
                )
                if total > 0
                else 0.0,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch performance metrics: {str(e)}"
        ) from e


@observability_router.get("/metrics/nodes")
async def get_node_performance_metrics(
    hours: int = Query(default=24, description="Hours of data to analyze"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get node-level performance metrics."""
    try:
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_time}}},
            {
                "$group": {
                    "_id": "$node_type",
                    "total_executions": {"$sum": 1},
                    "successful_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "failed_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                    },
                    "avg_duration_ms": {"$avg": "$duration_ms"},
                    "min_duration_ms": {"$min": "$duration_ms"},
                    "max_duration_ms": {"$max": "$duration_ms"},
                    "avg_input_size": {"$avg": "$input_size_bytes"},
                    "avg_output_size": {"$avg": "$output_size_bytes"},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        cursor = await db.node_executions.aggregate(pipeline)
        results = await cursor.to_list(None)

        node_metrics = {}
        for result in results:
            node_type = result["_id"]
            total = result.get("total_executions", 0)
            successful = result.get("successful_executions", 0)

            node_metrics[node_type] = {
                "total_executions": total,
                "successful_executions": successful,
                "failed_executions": result.get("failed_executions", 0),
                "success_rate": (successful / total * 100) if total > 0 else 0.0,
                "avg_duration_ms": round(result.get("avg_duration_ms", 0) or 0, 2),
                "min_duration_ms": result.get("min_duration_ms", 0),
                "max_duration_ms": result.get("max_duration_ms", 0),
                "avg_input_size_bytes": round(result.get("avg_input_size", 0) or 0, 2),
                "avg_output_size_bytes": round(
                    result.get("avg_output_size", 0) or 0, 2
                ),
            }

        return {
            "period_hours": hours,
            "cutoff_time": cutoff_time.isoformat(),
            "node_metrics": node_metrics,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch node metrics: {str(e)}"
        ) from e


@observability_router.get("/security/events")
async def get_security_events(
    limit: int = Query(default=100, description="Number of recent events to return"),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    severity: str | None = Query(default=None, description="Filter by severity level"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get recent security events with optional filtering."""
    try:
        ObservabilitySecurityMonitor(db)

        query_filter = {}
        if event_type:
            query_filter["event_type"] = event_type
        if severity:
            query_filter["severity"] = severity

        events = await db.security_events.find(
            query_filter, sort=[("timestamp", -1)], limit=limit
        ).to_list(None)

        for event in events:
            if "_id" in event:
                event["_id"] = str(event["_id"])

        return {
            "events": events,
            "count": len(events),
            "filters": {"event_type": event_type, "severity": severity},
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch security events: {str(e)}"
        ) from e


@observability_router.get("/security/metrics")
async def get_security_metrics(
    hours: int = Query(default=24, description="Hours of data to analyze"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get security metrics and trends."""
    try:
        security_monitor = ObservabilitySecurityMonitor(db)
        metrics = await security_monitor.get_security_metrics(hours)

        return {"security_metrics": metrics, "timestamp": datetime.now(UTC).isoformat()}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch security metrics: {str(e)}"
        ) from e


@observability_router.get("/security/trends")
async def get_security_trends(
    days: int = Query(default=7, description="Days of trend data to analyze"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get security trends over time."""
    try:
        security_monitor = ObservabilitySecurityMonitor(db)
        trends = await security_monitor.get_security_trends(days)

        return {"security_trends": trends, "timestamp": datetime.now(UTC).isoformat()}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch security trends: {str(e)}"
        ) from e


@observability_router.get("/dashboard/summary")
async def get_dashboard_summary(
    hours: int = Query(default=24, description="Hours of data for summary"),
    db: AsyncDatabase = Depends(get_db),
):
    """Get comprehensive dashboard summary for monitoring."""
    try:
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        exec_cursor = await db.agent_executions.aggregate(
            [
                {"$match": {"timestamp": {"$gte": cutoff_time}}},
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
                    }
                },
            ]
        )
        exec_stats = await exec_cursor.to_list(None)

        security_cursor = await db.security_events.aggregate(
            [
                {"$match": {"timestamp": {"$gte": cutoff_time}}},
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            ]
        )
        security_stats = await security_cursor.to_list(None)

        node_cursor = await db.node_executions.aggregate(
            [
                {"$match": {"timestamp": {"$gte": cutoff_time}}},
                {
                    "$group": {
                        "_id": "$node_type",
                        "avg_duration": {"$avg": "$duration_ms"},
                        "count": {"$sum": 1},
                    }
                },
            ]
        )
        node_stats = await node_cursor.to_list(None)

        exec_summary = (
            exec_stats[0]
            if exec_stats
            else {"total": 0, "successful": 0, "failed": 0, "avg_duration": 0}
        )

        security_summary = {}
        for stat in security_stats:
            security_summary[stat["_id"]] = stat["count"]

        node_summary = {}
        for stat in node_stats:
            node_summary[stat["_id"]] = {
                "avg_duration_ms": round(stat["avg_duration"] or 0, 2),
                "executions": stat["count"],
            }

        return {
            "period_hours": hours,
            "cutoff_time": cutoff_time.isoformat(),
            "summary": {
                "executions": {
                    "total": exec_summary["total"],
                    "successful": exec_summary["successful"],
                    "failed": exec_summary["failed"],
                    "success_rate": (
                        exec_summary["successful"] / exec_summary["total"] * 100
                    )
                    if exec_summary["total"] > 0
                    else 0,
                    "avg_duration_ms": round(
                        exec_summary.get("avg_duration", 0) or 0, 2
                    ),
                },
                "security": security_summary,
                "nodes": node_summary,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch dashboard summary: {str(e)}"
        ) from e


@observability_router.post("/alerts/performance")
async def create_performance_alert(
    alert_data: dict[str, Any], db: AsyncDatabase = Depends(get_db)
):
    """Create a performance alert (for integration with monitoring systems)."""
    try:
        alert_doc = {
            "timestamp": datetime.now(UTC),
            "alert_type": "performance",
            "severity": alert_data.get("severity", "medium"),
            "metric_name": alert_data.get("metric_name"),
            "current_value": alert_data.get("current_value"),
            "threshold_value": alert_data.get("threshold_value"),
            "description": alert_data.get("description"),
            "resolved": False,
            "metadata": alert_data.get("metadata", {}),
        }

        result = await db.performance_alerts.insert_one(alert_doc)

        return {
            "alert_id": str(result.inserted_id),
            "status": "created",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create alert: {str(e)}"
        ) from e


@observability_router.get("/system/health")
async def get_system_health(db: AsyncDatabase = Depends(get_db)):
    """Get overall system health status."""
    try:
        await db.command("ping")

        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)

        recent_executions = await db.agent_executions.count_documents(
            {"timestamp": {"$gte": one_hour_ago}}
        )

        recent_failures = await db.agent_executions.count_documents(
            {"timestamp": {"$gte": one_hour_ago}, "status": "failed"}
        )

        error_rate = (
            (recent_failures / recent_executions * 100) if recent_executions > 0 else 0
        )

        recent_security_events = await db.security_events.count_documents(
            {
                "timestamp": {"$gte": one_hour_ago},
                "severity": {"$in": ["high", "critical"]},
            }
        )

        health_status = "healthy"
        issues = []

        if error_rate > 10:  # More than 10% error rate
            health_status = "degraded"
            issues.append(f"High error rate: {error_rate:.1f}%")

        if recent_security_events > 5:  # More than 5 high-severity security events
            health_status = "degraded" if health_status == "healthy" else "unhealthy"
            issues.append(f"High security event count: {recent_security_events}")

        return {
            "status": health_status,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {
                "database_connectivity": "healthy",
                "error_rate": {
                    "status": "healthy"
                    if error_rate <= 5
                    else "degraded"
                    if error_rate <= 10
                    else "unhealthy",
                    "value": round(error_rate, 2),
                    "threshold": 10.0,
                },
                "security_events": {
                    "status": "healthy" if recent_security_events <= 5 else "degraded",
                    "value": recent_security_events,
                    "threshold": 5,
                },
            },
            "issues": issues,
            "metrics": {
                "executions_last_hour": recent_executions,
                "failures_last_hour": recent_failures,
                "security_events_last_hour": recent_security_events,
            },
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "error": str(e),
            "checks": {"database_connectivity": "failed"},
        }
