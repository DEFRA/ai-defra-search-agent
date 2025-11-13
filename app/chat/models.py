import dataclasses
import uuid

@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str
    model: str = "Unknown"


@dataclasses.dataclass(frozen=True)
class Conversation:
    conversation_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = dataclasses.field(default_factory=list)
