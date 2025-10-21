import dataclasses
import datetime


@dataclasses.dataclass(frozen=True)
class KnowledgeDocument:
    content: str
    snapshot_id: str
    source_id: str
    metadata: dict[str, any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclasses.dataclass(frozen=True)
class StageTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime.datetime = dataclasses.field(default_factory=lambda: datetime.datetime.now(tz=datetime.UTC))
    model: str = "Unknown"
    stage_name: str = "Unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ConversationHistory:
    def __init__(self):
        self.messages: list[ChatMessage] = []

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def get_history(self, limit=10) -> list[ChatMessage]:
        return self.messages[-limit:]

