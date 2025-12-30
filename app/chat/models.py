import dataclasses
import datetime
import uuid
from typing import Any, Literal, Optional

__all__ = [
    "AgentRequest",
    "AssistantMessage",
    "Conversation",
    "ConversationNotFoundError",
    "Message",
    "TokenUsage",
    "UserMessage",
]


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclasses.dataclass(frozen=True)
class AgentRequest:
    question: str
    model_id: str
    conversation: Optional[list["Message"]] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class Message:
    role: str
    content: str
    model_id: str
    model_name: str
    timestamp: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [{"text": self.content}]}


@dataclasses.dataclass(frozen=True, kw_only=True)
class UserMessage(Message):
    role: Literal["user"] = "user"


@dataclasses.dataclass(frozen=True, kw_only=True)
class AssistantMessage(Message):
    usage: TokenUsage
    role: Literal["assistant"] = "assistant"


@dataclasses.dataclass
class Conversation:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    messages: list[Message] = dataclasses.field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


class ConversationNotFoundError(Exception):
    pass
