from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class ConversationHistory:
    conversation_id: UUID
    messages: list[ChatMessage]

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))


class ConversationNotFoundError(Exception):
    """Exception raised when a conversation is not found."""
