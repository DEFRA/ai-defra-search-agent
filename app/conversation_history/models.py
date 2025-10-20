from dataclasses import dataclass, field
from uuid import UUID

from app.v2_chat.models import StageTokenUsage


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class ConversationHistory:
    conversation_id: UUID
    messages: list[ChatMessage] = field(default_factory=list)
    token_usage: list[StageTokenUsage] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))


class ConversationNotFoundError(Exception):
    """Exception raised when a conversation is not found."""
