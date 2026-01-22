import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from botocore.exceptions import ClientError
from fastapi import status

from app.chat import dependencies, models
from app.common.event_broker import get_event_broker

logger = logging.getLogger(__name__)

# Worker configuration constants
SQS_MAX_MESSAGES_PER_POLL = 1
SQS_LONG_POLL_WAIT_SECONDS = 20
WORKER_ERROR_RETRY_DELAY_SECONDS = 5

# Worker health tracking
_last_heartbeat: datetime | None = None


def get_last_heartbeat() -> datetime | None:
    """Get the last worker heartbeat timestamp."""
    return _last_heartbeat


async def _update_message_failed(
    conversation_repository,
    event_broker,
    conversation_id: uuid.UUID | None,
    message_id: uuid.UUID,
    error_message: str,
    error_code: int,
) -> None:
    """Update message status to FAILED and publish failure event."""
    if conversation_id:
        await conversation_repository.update_message_status(
            conversation_id=conversation_id,
            message_id=message_id,
            status=models.MessageStatus.FAILED,
            error_message=error_message,
            error_code=error_code,
        )
    await event_broker.publish(
        str(message_id),
        {
            "status": models.MessageStatus.FAILED.value,
            "message_id": str(message_id),
            "error_message": error_message,
            "error_code": error_code,
        },
    )


async def process_job_message(
    message: dict, chat_service, conversation_repository, sqs_client
) -> None:
    body = json.loads(message["Body"])
    conversation_id = (
        uuid.UUID(body["conversation_id"]) if body.get("conversation_id") else None
    )
    message_id = uuid.UUID(body["message_id"])
    question = body["question"]
    model_id = body["model_id"]
    receipt_handle = message["ReceiptHandle"]
    event_broker = get_event_broker()

    try:
        # Update message status to PROCESSING
        if conversation_id:
            await conversation_repository.update_message_status(
                conversation_id=conversation_id,
                message_id=message_id,
                status=models.MessageStatus.PROCESSING,
            )
        await event_broker.publish(
            str(message_id),
            {
                "status": models.MessageStatus.PROCESSING.value,
                "message_id": str(message_id),
            },
        )

        # Execute chat
        conversation = await chat_service.execute_chat(
            question=question,
            model_id=model_id,
            conversation_id=conversation_id,
        )

        # Update message status to COMPLETED
        await conversation_repository.update_message_status(
            conversation_id=conversation.id,
            message_id=message_id,
            status=models.MessageStatus.COMPLETED,
        )

        # Publish completion event with full conversation
        result = {
            "conversation_id": str(conversation.id),
            "messages": [
                {
                    "message_id": str(msg.message_id),
                    "role": msg.role,
                    "content": msg.content,
                    "model_name": msg.model_name,
                    "model_id": msg.model_id,
                    "status": msg.status.value,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in conversation.messages
            ],
        }

        await event_broker.publish(
            str(message_id),
            {
                "status": models.MessageStatus.COMPLETED.value,
                "message_id": str(message_id),
                "result": result,
            },
        )
    except models.ConversationNotFoundError as e:
        await _update_message_failed(
            conversation_repository,
            event_broker,
            conversation_id,
            message_id,
            f"Conversation not found: {e}",
            status.HTTP_404_NOT_FOUND,
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

        await _update_message_failed(
            conversation_repository,
            event_broker,
            conversation_id,
            message_id,
            f"{error_type}: {error_msg}",
            error_code,
        )
    except Exception as e:
        logger.exception("Failed to process message %s", message_id)
        await _update_message_failed(
            conversation_repository,
            event_broker,
            conversation_id,
            message_id,
            str(e),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        await sqs_client.delete_message(receipt_handle)


async def run_worker():
    global _last_heartbeat
    logger.info("Starting chat worker")

    (
        chat_service,
        conversation_repository,
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
                        message, chat_service, conversation_repository, sqs_client
                    )
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_DELAY_SECONDS)


def main():
    """Entry point for running worker as a module."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
