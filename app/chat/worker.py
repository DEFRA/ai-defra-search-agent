import asyncio
import json
import logging

from botocore.exceptions import ClientError

from app.chat import dependencies, job_models, models
from app.common.event_broker import get_event_broker

logger = logging.getLogger(__name__)


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
            error_code=404,
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.FAILED.value,
                "job_id": job_id,
                "error_message": f"Conversation not found: {e}",
                "error_code": 404,
            },
        )
    except ClientError as e:
        # Map AWS ClientError to appropriate HTTP status codes
        error_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        error_type = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        # Map specific AWS error types to HTTP codes
        if error_type == "ThrottlingException":
            error_code = 429
        elif error_type == "ServiceUnavailableException":
            error_code = 503
        elif error_type in ["InternalServerException", "InternalFailure"]:
            error_code = 500

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
            error_code=500,
        )
        await event_broker.publish(
            job_id,
            {
                "status": job_models.JobStatus.FAILED.value,
                "job_id": job_id,
                "error_message": str(e),
                "error_code": 500,
            },
        )
    finally:
        await sqs_client.delete_message(receipt_handle)


async def run_worker():
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
                    max_messages=1, wait_time=20
                )

                for message in messages:
                    await process_job_message(
                        message, chat_service, job_repository, sqs_client
                    )
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(5)


def main():
    """Entry point for running worker as a module."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
