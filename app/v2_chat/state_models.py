import dataclasses

from app.v2_chat import models


@dataclasses.dataclass
class InputState:
    question: str


@dataclasses.dataclass(frozen=True)
class OutputState:
    final_answer: str
    context: list[models.KnowledgeDocument] = dataclasses.field(default_factory=list)
    token_usage: list[models.StageTokenUsage] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ChatState:
    question: str
    answer: str = ""
    final_answer: str = ""
    candidate_documents: list[models.KnowledgeDocument] = dataclasses.field(default_factory=list)
    context: list[models.KnowledgeDocument] = dataclasses.field(default_factory=list)
    conversation_history: list[models.ChatMessage] = dataclasses.field(default_factory=list)
    token_usage: list[models.StageTokenUsage] = dataclasses.field(default_factory=list)
