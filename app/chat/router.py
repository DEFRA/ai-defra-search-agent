import asyncio
import json
import logging
import uuid

import fastapi
from fastapi import status
from sse_starlette.sse import EventSourceResponse

from app.chat import api_schemas, dependencies, job_models
from app.common.event_broker import get_event_broker
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    status_code=status.HTTP_200_OK,
    summary="Send a message to the chatbot with SSE streaming",
    description="Sends a user question to the specified model and streams the response status updates via Server-Sent Events.",
    responses={
        200: {"description": "SSE stream of job status updates"},
        400: {
            "description": "Bad request - unsupported model ID, invalid request data, or AWS Bedrock validation error"
        },
    },
)
async def chat(
    request: api_schemas.ChatRequest,
    job_repository=fastapi.Depends(dependencies.get_job_repository),
    sqs_client=fastapi.Depends(dependencies.get_sqs_client),
    model_resolution_service=fastapi.Depends(dependencies.get_model_resolution_service),
):
    # Validate model immediately
    try:
        model_resolution_service.resolve_model(request.model_id)
    except UnsupportedModelError as e:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from None

    # Create job
    job = job_models.ChatJob(
        conversation_id=request.conversation_id,
        question=request.question,
        model_id=request.model_id,
    )

    await job_repository.create(job)

    # Queue for processing
    async with sqs_client:
        await sqs_client.send_message(
            {
                "job_id": str(job.job_id),
                "conversation_id": str(request.conversation_id)
                if request.conversation_id
                else None,
                "question": request.question,
                "model_id": request.model_id,
            }
        )

    # Stream job status updates via SSE
    async def event_generator():
        event_broker = get_event_broker()
        queue = await event_broker.subscribe(str(job.job_id))

        try:
            # Send initial queued status
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "job_id": str(job.job_id),
                        "status": job_models.JobStatus.QUEUED.value,
                    }
                ),
            }

            # Stream updates from event broker
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield {
                        "event": "status",
                        "data": json.dumps(event),
                    }

                    # End stream after terminal status
                    if event.get("status") in [
                        job_models.JobStatus.COMPLETED.value,
                        job_models.JobStatus.FAILED.value,
                    ]:
                        break

                except TimeoutError:
                    # Send keepalive comment
                    yield {"event": "keepalive", "data": ""}

        except asyncio.CancelledError:
            logger.info("SSE connection cancelled for job %s", job.job_id)
        finally:
            await event_broker.unsubscribe(str(job.job_id), queue)

    return EventSourceResponse(event_generator(), ping=None)


@router.get(
    "/jobs/{job_id}",
    summary="Get status by job ID",
    description="Retrieve the current status and details of a chat job.",
)
async def get_job(
    job_id: uuid.UUID,
    job_repository=fastapi.Depends(dependencies.get_job_repository),
):
    job = await job_repository.get(job_id)
    if job is None:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return job
