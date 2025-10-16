from dataclasses import dataclass, field

from app.v2_chat.models import ChatMessage, KnowledgeDocument


@dataclass
class InputState:
    question: str


@dataclass(frozen=True)
class OutputState:
    answer: str
    context: list[KnowledgeDocument] = field(default_factory=list)


@dataclass
class ChatState:
    question: str
    answer: str = ""
    candidate_documents: list[KnowledgeDocument] = field(default_factory=list)
    context: list[KnowledgeDocument] = field(default_factory=list)
    conversation_history: list[ChatMessage] = field(default_factory=list)
