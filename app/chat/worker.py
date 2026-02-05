"""Background worker for processing chat messages from SQS.

This module implements the long-running worker loop that polls SQS for
queued chat messages, dispatches them to the `ChatService` for processing,
and updates message status in the conversation repository. It also
exposes a small health helper to report the worker's last heartbeat time.
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from botocore.exceptions import ClientError
from fastapi import status

from app import config
from app.chat import dependencies, models

logger = logging.getLogger(__name__)

_last_heartbeat: datetime | None = None


def get_last_heartbeat() -> datetime | None:
    """Return the timestamp of the last successful SQS poll.

    Returns:
        datetime | None: UTC timestamp of the last worker heartbeat, or
        ``None`` if the worker has not yet completed a poll.
    """

    return _last_heartbeat


async def _update_message_failed(
    conversation_repository,
    conversation_id: uuid.UUID | None,
    message_id: uuid.UUID,
    error_message: str,
    error_code: int,
) -> None:
    """Mark a message as failed in the conversation repository.

    The helper only updates the repository when a `conversation_id` is
    available (some SQS jobs may not carry a conversation reference).

    Args:
        conversation_repository: Repository implementing `update_message_status`.
        conversation_id: Optional UUID of the conversation to update.
        message_id: UUID of the failed message.
        error_message: Human readable error message to store.
        error_code: HTTP-like error code to record.
    """

    if conversation_id:
        await conversation_repository.update_message_status(
            conversation_id=conversation_id,
            message_id=message_id,
            status=models.MessageStatus.FAILED,
            error_message=error_message,
            error_code=error_code,
        )


async def _claim_or_skip(
    conversation_repository,
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
) -> bool:
    """Attempt to reserve a queued message (set to PROCESSING) or acknowledge & skip it.

    This performs a single repository update that checks the current
    status and sets it to `processing`. If the reservation fails, the helper inspects the
    current message status and acknowledges the SQS message when it's
    already completed, missing, or otherwise being handled by another
    worker.

    Returns True when processing should be skipped (message acknowledged or
    another worker is handling it). Returns False when the caller should
    proceed to process the message.
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
        # Already processed successfully
        return True

    # Not queued and not completed; skip processing to avoid duplicate work.
    logger.info("Skipping message %s with status %s", message_id, current_status)
    return True


async def process_job_message(
    message: dict, chat_service, conversation_repository, sqs_client
) -> None:
    """Process a single SQS job message.

    This function is responsible for:
    - Decoding the SQS message body
    - Transitioning the message status to PROCESSING
    - Calling the `chat_service.execute_chat` to produce a conversation
      result
    - Updating the message status to COMPLETED or FAILED depending on
      errors
    - Ensuring the message is deleted from SQS after processing

    Args:
        message: Raw SQS message dictionary as returned by boto3.
        chat_service: Service implementing `execute_chat`.
        conversation_repository: Repository managing conversations/messages.
        sqs_client: SQS client wrapper with `delete_message`.
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
        # Attempt to reserve the queued message for processing by checking
        # the current status and setting it to `processing` in a single
        # repository update. If the reservation fails, the helper will
        # acknowledge the message and indicate that processing should be
        # skipped.
        if conversation_id:
            should_skip = await _claim_or_skip(
                conversation_repository, conversation_id, message_id
            )
            if should_skip:
                return

        # Execute chat
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
            conversation_id,
            message_id,
            f"{error_type}: {error_msg}",
            error_code,
        )
    except Exception as e:
        logger.exception("Failed to process message %s", message_id)
        await _update_message_failed(
            conversation_repository,
            conversation_id,
            message_id,
            str(e),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        await asyncio.to_thread(sqs_client.delete_message, receipt_handle)


async def run_worker():
    global _last_heartbeat
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

                _last_heartbeat = datetime.now(UTC)

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
