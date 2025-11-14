import dataclasses
import uuid


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str
    model: str | None = None


@dataclasses.dataclass
class Conversation:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    messages: list[Message] = dataclasses.field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


class ConversationNotFoundError(Exception):
    pass
