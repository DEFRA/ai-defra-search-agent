import dataclasses
import datetime
import enum
import uuid


class WasHelpfulRating(str, enum.Enum):
    """Enum for feedback helpfulness ratings."""

    VERY_USEFUL = "very-useful"
    USEFUL = "useful"
    NEITHER = "neither"
    NOT_USEFUL = "not-useful"
    NOT_AT_ALL_USEFUL = "not-at-all-useful"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Feedback:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID | None = None
    was_helpful: WasHelpfulRating
    comment: str | None = None
    timestamp: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
