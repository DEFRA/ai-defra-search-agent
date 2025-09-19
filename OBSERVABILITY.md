# MongoDB-Based Observability for LangGraph Agents

This document explains how to implement comprehensive observability for your LangGraph agents using MongoDB and FastAPI endpoints, without requiring OpenTelemetry.

## Architecture Overview

The observability solution consists of several key components:

1. **MongoDB Collections** - Store execution traces, metrics, and security events
2. **Observability Service** - Tracks and stores execution data
3. **Enhanced Security Monitor** - Stores security events with correlation
4. **FastAPI Endpoints** - Expose observability data through REST APIs
5. **Real-time Metrics Collector** - Provides live monitoring data

## MongoDB Schema

### Collections Created

- `agent_executions` - Complete agent execution traces
- `node_executions` - Individual LangGraph node executions
- `security_events` - Security monitoring events
- `system_metrics` - Aggregated system metrics
- `performance_alerts` - Performance threshold violations
- `live_metrics` - Real-time metrics for dashboards

## Key Features

### 1. Complete Execution Tracing
Every agent execution is tracked from start to finish:
- Unique execution ID for correlation
- Timing metrics (start, end, duration)
- Input/output validation status
- Token usage and cost tracking
- Error handling and stack traces

### 2. Node-Level Performance Tracking
Individual LangGraph nodes are monitored:
- Retrieve, Grade Documents, Generate performance
- Input/output data sizes
- Node-specific metrics (document counts, tokens)
- Error rates per node type

### 3. Security Event Correlation
Security events are linked to executions:
- Injection attempts
- Off-topic queries
- Validation failures
- System errors
- All correlated with execution context

### 4. Live Monitoring Endpoints
FastAPI endpoints provide real-time access to:
- Recent executions and performance metrics
- Security events and trends
- System health status
- Node-level performance data
- Dashboard summary data

## Usage Examples

### 1. Basic Integration

Update your existing RAG function:

```python
# Instead of using the original run_rag_llm
from app.lib.enhanced_langgraph_rag import run_rag_llm_with_observability

# Use the enhanced version
async def chat_endpoint(query: str):
    result = await run_rag_llm_with_observability(query)
    return result
```

### 2. FastAPI Router Integration

Add observability endpoints to your main app:

```python
from fastapi import FastAPI
from app.observability.router import observability_router

app = FastAPI()
app.include_router(observability_router)
```

### 3. Accessing Observability Data

```python
# Get recent executions
GET /observability/executions/recent?limit=50

# Get detailed execution trace
GET /observability/executions/{execution_id}

# Get performance metrics
GET /observability/metrics/performance?hours=24

# Get security events
GET /observability/security/events?limit=100

# Get system health
GET /observability/system/health
```

## Available Endpoints

### Execution Monitoring
- `GET /observability/executions/recent` - Recent executions
- `GET /observability/executions/{id}` - Execution details
- `GET /observability/metrics/performance` - Performance metrics
- `GET /observability/metrics/nodes` - Node-level metrics

### Security Monitoring
- `GET /observability/security/events` - Security events
- `GET /observability/security/metrics` - Security metrics
- `GET /observability/security/trends` - Security trends

### System Health
- `GET /observability/system/health` - Overall system health
- `GET /observability/health` - Basic health check
- `GET /observability/dashboard/summary` - Dashboard summary

## Example Responses

### Performance Metrics
```json
{
  "period_hours": 24,
  "metrics": {
    "total_executions": 1250,
    "successful_executions": 1198,
    "failed_executions": 52,
    "success_rate": 95.84,
    "avg_duration_ms": 2450.5,
    "avg_documents_retrieved": 4.2,
    "validation_failure_rate": 2.1
  }
}
```

### Execution Details
```json
{
  "execution": {
    "execution_id": "exec_123456",
    "timestamp": "2025-09-18T10:30:00Z",
    "query_hash": "a1b2c3d4e5f6",
    "status": "completed",
    "total_duration_ms": 2100,
    "answer": "Based on the documents...",
    "source_documents_count": 3,
    "input_validation_passed": true,
    "output_validation_passed": true,
    "token_usage": {
      "input_tokens": 150,
      "output_tokens": 75,
      "total_tokens": 225
    }
  },
  "nodes": [
    {
      "node_name": "retrieve",
      "node_type": "retrieve",
      "status": "completed",
      "duration_ms": 850,
      "retrieved_docs_count": 5
    }
  ]
}
```

### Security Events
```json
{
  "events": [
    {
      "event_type": "injection_attempt",
      "timestamp": "2025-09-18T10:25:00Z",
      "severity": "high",
      "details": "Potential prompt injection detected",
      "metadata": {
        "execution_id": "exec_123456",
        "pattern_matched": "ignore_instructions"
      }
    }
  ]
}
```

## Dashboard Integration

The endpoints provide all data needed for monitoring dashboards:

### Real-time Metrics
- Current success/failure rates
- Average response times
- Document retrieval statistics
- Security event counts

### Historical Trends
- Hourly/daily performance trends
- Security event patterns
- Error rate trends
- Usage patterns

### Health Monitoring
- System health status
- Database connectivity
- Error rate thresholds
- Security event thresholds

## Benefits of This Approach

### 1. No External Dependencies
- Uses existing MongoDB infrastructure
- No OpenTelemetry setup required
- Leverages FastAPI for endpoints

### 2. Complete Visibility
- Full execution traces
- Node-level performance
- Security event correlation
- Real-time monitoring

### 3. Easy Integration
- Drop-in replacement for existing function
- FastAPI router integration
- Backward compatible responses

### 4. Flexible Querying
- MongoDB's powerful aggregation
- Custom time ranges
- Filtering capabilities
- Real-time data access

### 5. Production Ready
- Error handling
- Performance optimized
- Scalable architecture
- Security focused

## Performance Considerations

- **MongoDB Indexing**: Create indexes on timestamp, execution_id, and status fields
- **Data Retention**: Implement data retention policies for old execution data
- **Aggregation Caching**: Cache frequently accessed metrics
- **Async Operations**: All database operations are async for better performance

## Monitoring and Alerting

The system provides endpoints for:
- Performance threshold violations
- Security event spikes
- System health degradation
- Error rate increases

These can be integrated with monitoring systems like Prometheus, Grafana, or custom alerting solutions.

This comprehensive observability solution gives you complete visibility into your LangGraph agent performance without requiring external tracing infrastructure.
