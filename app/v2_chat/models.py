from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeDocument:
    content: str
    snapshot_id: str
    source_id: str
    metadata: dict[str, any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ConversationHistory:
    def __init__(self):
        self.messages: list[ChatMessage] = []

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def get_history(self, limit=10) -> list[ChatMessage]:
        return self.messages[-limit:]

