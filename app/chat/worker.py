"""Background worker for processing chat messages from SQS.

Polls SQS for queued messages, dispatches to ChatService, and updates message status.
"""

import asyncio
import json
import logging
import uuid

from botocore.exceptions import ClientError
from fastapi import status

from app import config
from app.chat import dependencies, models

logger = logging.getLogger(__name__)


def _map_aws_error_to_status_code(error_type: str) -> int:
    """Map AWS error type to HTTP status code for logging."""
    match error_type:
        case "ThrottlingException":
            return status.HTTP_429_TOO_MANY_REQUESTS
        case "ServiceUnavailableException":
            return status.HTTP_503_SERVICE_UNAVAILABLE
        case "InternalServerException" | "InternalFailure":
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        case _:
            return status.HTTP_500_INTERNAL_SERVER_ERROR


async def _update_message_failed(
    conversation_repository,
    conversation_id: uuid.UUID | None,
    message_id: uuid.UUID,
    error_message: str,
) -> None:
    """Mark a message as failed in the repository.

    Only updates when conversation_id is available.
    """

    if conversation_id:
        await conversation_repository.update_message_status(
            conversation_id=conversation_id,
            message_id=message_id,
            status=models.MessageStatus.FAILED,
            error_message=error_message,
        )


async def _claim_or_skip(
    conversation_repository,
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
) -> bool:
    """Atomically claim a queued message by setting status to PROCESSING.

    Returns True if processing should be skipped (already completed or being handled).
    """

    claimed = await conversation_repository.claim_message(
        conversation_id=conversation_id, message_id=message_id
    )
    if claimed:
        return False

    current_status = await conversation_repository.get_message_status(
        conversation_id=conversation_id, message_id=message_id
    )
    if current_status is None:
        logger.warning(
            "No DB record found for message %s; acknowledged and skipping", message_id
        )
        return True

    if current_status == models.MessageStatus.COMPLETED:
        return True

    logger.info("Skipping message %s with status %s", message_id, current_status)
    return True


async def process_job_message(
    message: dict, chat_service, conversation_repository, sqs_client
) -> None:
    """Process a single SQS message from queue to completion.

    Decodes message, claims it, executes chat, updates status, and deletes from SQS.
    Handles errors by marking message as FAILED with appropriate error details.
    """

    body = json.loads(message["Body"])
    conversation_id = (
        uuid.UUID(body["conversation_id"]) if body.get("conversation_id") else None
    )
    message_id = uuid.UUID(body["message_id"])
    question = body["question"]
    model_id = body["model_id"]
    receipt_handle = message["ReceiptHandle"]

    try:
        if conversation_id:
            should_skip = await _claim_or_skip(
                conversation_repository, conversation_id, message_id
            )
            if should_skip:
                return

        conversation = await chat_service.execute_chat(
            question=question,
            model_id=model_id,
            message_id=message_id,
            conversation_id=conversation_id,
        )

        await conversation_repository.update_message_status(
            conversation_id=conversation.id,
            message_id=message_id,
            status=models.MessageStatus.COMPLETED,
        )

    except models.ConversationNotFoundError as e:
        await _update_message_failed(
            conversation_repository,
            conversation_id,
            message_id,
            f"Conversation not found: {e}",
        )
    except ClientError as e:
        error_type = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        error_code = _map_aws_error_to_status_code(error_type)

        logger.error(
            "AWS ClientError processing message %s: %s (HTTP %d)",
            message_id,
            error_type,
            error_code,
        )
        await _update_message_failed(
            conversation_repository,
            conversation_id,
            message_id,
            f"{error_type}: {error_msg}",
        )
    except Exception as e:
        logger.exception("Failed to process message %s", message_id)
        await _update_message_failed(
            conversation_repository,
            conversation_id,
            message_id,
            str(e),
        )
    finally:
        await asyncio.to_thread(sqs_client.delete_message, receipt_handle)


async def run_worker():
    """Main worker loop that polls SQS and processes chat messages.

    Runs indefinitely until application shutdown.
    """
    logger.info("Starting chat worker")

    (
        chat_service,
        conversation_repository,
        sqs_client,
    ) = await dependencies.initialize_worker_services()

    with sqs_client:
        while True:
            try:
                messages = await asyncio.to_thread(
                    sqs_client.receive_messages,
                    max_messages=config.config.chat_queue.batch_size,
                    wait_time=config.config.chat_queue.wait_time,
                )

                for message in messages:
                    await process_job_message(
                        message, chat_service, conversation_repository, sqs_client
                    )
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(config.config.chat_queue.polling_interval)


def main():
    """Entry point for running worker as a module."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
