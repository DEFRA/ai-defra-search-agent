import dataclasses
import datetime
import enum
import uuid
from typing import Any, Literal

__all__ = [
    "AgentRequest",
    "AssistantMessage",
    "Conversation",
    "ConversationNotFoundError",
    "Message",
    "MessageStatus",
    "Source",
    "TokenUsage",
    "UserMessage",
]


class MessageStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclasses.dataclass(frozen=True)
class AgentRequest:
    question: str
    model_id: str
    conversation: list["Message"] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class Message:
    role: str
    content: str
    model_id: str
    model_name: str
    message_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    timestamp: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [{"text": self.content}]}


@dataclasses.dataclass(frozen=True, kw_only=True)
class UserMessage(Message):
    role: Literal["user"] = "user"
    status: MessageStatus = MessageStatus.COMPLETED
    error_message: str | None = None


@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    location: str
    snippet: str
    score: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class AssistantMessage(Message):
    usage: TokenUsage
    sources: list[Source] = dataclasses.field(default_factory=list)
    role: Literal["assistant"] = "assistant"


@dataclasses.dataclass
class Conversation:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    messages: list[Message] = dataclasses.field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


class ConversationNotFoundError(Exception):
    pass
