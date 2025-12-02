import dataclasses
import datetime
import uuid


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str
    model_id: str | None = None
    usage: TokenUsage | None = None
    created_at: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )


@dataclasses.dataclass
class Conversation:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    messages: list[Message] = dataclasses.field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


class ConversationNotFoundError(Exception):
    pass


class UnsupportedModelError(Exception):
    pass
