import dataclasses
import uuid

from app.v2_chat import models as chat_models


@dataclasses.dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclasses.dataclass
class ConversationHistory:
    conversation_id: uuid.UUID
    messages: list[ChatMessage] = dataclasses.field(default_factory=list)
    token_usage: list[chat_models.StageTokenUsage] = dataclasses.field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))


class ConversationNotFoundError(Exception):
    """Exception raised when a conversation is not found."""
