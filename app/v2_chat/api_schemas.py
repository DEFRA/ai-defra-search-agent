from app.v2_chat.models import KnowledgeDocument
from pydantic import BaseModel, Field
from uuid import uuid4

class ChatRequest(BaseModel):
    question: str = Field(
        description="The question to ask the model",
        examples=[
            "What ethical consideration do we need to make sure we cover using AI?"
        ],
    )
    conversation_id: str | None = Field(
        default_factory=lambda: str(uuid4()),
        description="UUIDv4 identifier for the conversation",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        description="The answer from the model",
        examples=[
            "When using AI, it is important to consider the ethical implications of its use, including issues such as bias, privacy, and accountability."
        ],
    )
    conversation_id: str = Field(
        description="UUIDv4 identifier for the conversation",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    context_documents: list[dict] = Field(
        description="The documents used to generate the answer",
    )
