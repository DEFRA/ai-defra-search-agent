"""In-memory event broker for publishing job status updates."""

import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EventBroker:
    """Simple in-memory event broker for pub/sub of job status updates."""

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to updates for a specific job.

        Args:
            job_id: The job ID to subscribe to

        Returns:
            An asyncio.Queue that will receive job status updates
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers[job_id].append(queue)
        logger.debug("Subscribed to job %s", job_id)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from updates for a specific job.

        Args:
            job_id: The job ID to unsubscribe from
            queue: The queue to remove
        """
        async with self._lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(queue)
                    if not self._subscribers[job_id]:
                        del self._subscribers[job_id]
                    logger.debug("Unsubscribed from job %s", job_id)
                except ValueError:
                    pass

    async def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Publish an event to all subscribers of a job.

        Args:
            job_id: The job ID to publish to
            event: The event data to publish
        """
        async with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            logger.debug(
                "Publishing event to %d subscribers for job %s",
                len(subscribers),
                job_id,
            )

            for queue in subscribers:
                try:
                    await queue.put(event)
                except Exception as e:
                    logger.error("Failed to publish event to subscriber: %s", e)


# Global singleton instance
_broker: EventBroker | None = None


def get_event_broker() -> EventBroker:
    """Get the global event broker instance."""
    global _broker
    if _broker is None:
        _broker = EventBroker()
    return _broker
