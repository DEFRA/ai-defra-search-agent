import dataclasses
import datetime
import uuid


@dataclasses.dataclass(frozen=True, kw_only=True)
class Feedback:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID | None = None
    was_helpful: bool
    comment: str | None = None
    timestamp: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
