import logging
import uuid

import fastapi
from fastapi import status

from app.chat import api_schemas, dependencies, job_models
from app.models import UnsupportedModelError

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post(
    "/chat",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a message to the chatbot",
    description="Sends a user question to the specified model and retrieves the response along with the conversation history.",
    responses={
        202: {"description": "Request accepted and queued for processing"},
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

    # Create
    job = job_models.ChatJob(
        conversation_id=request.conversation_id,
        question=request.question,
        model_id=request.model_id,
    )

    await job_repository.create(job)

    # Queue for processing
    sqs_client.send_message(
        {
            "job_id": str(job.job_id),
            "conversation_id": str(request.conversation_id)
            if request.conversation_id
            else None,
            "question": request.question,
            "model_id": request.model_id,
        }
    )

    return {"job_id": job.job_id, "status": job.status}


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
