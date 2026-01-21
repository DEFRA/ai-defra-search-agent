import asyncio
import json
import logging
from datetime import UTC, datetime

from botocore.exceptions import ClientError
from fastapi import status

from app.chat import dependencies, job_models, models
from app.common.event_broker import get_event_broker

logger = logging.getLogger(__name__)

# Worker configuration constants
SQS_MAX_MESSAGES_PER_POLL = 1
SQS_LONG_POLL_WAIT_SECONDS = 20
WORKER_ERROR_RETRY_DELAY_SECONDS = 5

# Worker health tracking
_last_heartbeat: datetime | None = None


async def process_job_message(
    message: dict, chat_service, job_repository, sqs_client
) -> None:
    body = json.loads(message["Body"])
    job_id = body["job_id"]
    receipt_handle = message["ReceiptHandle"]
    event_broker = get_event_broker()

    try:
        await job_repository.update(
            job_id=job_id, status=job_models.JobStatus.PROCESSING
        )
        await event_broker.publish(
            job_id, {"status": job_models.JobStatus.PROCESSING.value, "job_id": job_id}
        )

        conversation = await chat_service.execute_chat(
            question=body["question"],
            model_id=body["model_id"],
            conversation_id=body.get("conversation_id"),
        )

        result = {
            "conversation_id": str(conversation.id),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "model_name": msg.model_name,
                    "model_id": msg.model_id,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in conversation.messages
            ],
        }

        await job_repository.update(
            job_id, status=job_models.JobStatus.COMPLETED, result=result
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.COMPLETED.value,
                "job_id": job_id,
                "result": result,
            },
        )
    except models.ConversationNotFoundError as e:
        await job_repository.update(
            job_id,
            status=job_models.JobStatus.FAILED,
            error_message=f"Conversation not found: {e}",
            error_code=status.HTTP_404_NOT_FOUND,
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.FAILED.value,
                "job_id": job_id,
                "error_message": f"Conversation not found: {e}",
                "error_code": status.HTTP_404_NOT_FOUND,
            },
        )
    except ClientError as e:
        # Map AWS ClientError to appropriate HTTP status codes
        error_code = e.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        error_type = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        # Map specific AWS error types to HTTP codes
        if error_type == "ThrottlingException":
            error_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif error_type == "ServiceUnavailableException":
            error_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif error_type in ["InternalServerException", "InternalFailure"]:
            error_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        await job_repository.update(
            job_id,
            status=job_models.JobStatus.FAILED,
            error_message=f"{error_type}: {error_msg}",
            error_code=error_code,
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.FAILED.value,
                "job_id": job_id,
                "error_message": f"{error_type}: {error_msg}",
                "error_code": error_code,
            },
        )
    except Exception as e:
        logger.exception("Failed to process job %s", job_id)
        await job_repository.update(
            job_id,
            status=job_models.JobStatus.FAILED,
            error_message=str(e),
            error_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.FAILED.value,
                "job_id": job_id,
                "error_message": str(e),
                "error_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            },
        )
    finally:
        await sqs_client.delete_message(receipt_handle)


# Worker health tracking
_last_heartbeat: datetime | None = None


def get_last_heartbeat() -> datetime | None:
    """Get the last worker heartbeat timestamp."""
    return _last_heartbeat


async def run_worker():
    global _last_heartbeat
    logger.info("Starting chat worker")

    (
        chat_service,
        job_repository,
        sqs_client,
    ) = await dependencies.initialize_worker_services()

    async with sqs_client:
        while True:
            try:
                messages = await sqs_client.receive_messages(
                    max_messages=SQS_MAX_MESSAGES_PER_POLL,
                    wait_time=SQS_LONG_POLL_WAIT_SECONDS,
                )

                # Update heartbeat after successful poll (even if no messages)
                _last_heartbeat = datetime.now(UTC)

                for message in messages:
                    await process_job_message(
                        message, chat_service, job_repository, sqs_client
                    )
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_DELAY_SECONDS)


def main():
    """Entry point for running worker as a module."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
