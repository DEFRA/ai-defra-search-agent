
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        description="The question to ask the model",
        examples=[
            "What ethical consideration do we need to make sure we cover using AI?"
        ],
    )


class ContextDocumentResponse(BaseModel):
    content: str = Field(description="Content of matched document content")
    title: str = Field(description="Title of document or knowledge source")
    snapshot_id: str = Field(
        description="Internal identifier for parent knowledge snapshot",
        serialization_alias="snapshotId",
    )
    source_id: str = Field(
        description="Internal identifier for source document",
        serialization_alias="sourceId",
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        description="The answer from the model",
        examples=[
            "When using AI, it is important to consider the ethical implications of its use, including issues such as bias, privacy, and accountability."
        ],
    )
    conversation_id: UUID = Field(
        description="UUIDv4 identifier for the conversation",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
        serialization_alias="conversationId",
    )
    context_documents: list[ContextDocumentResponse] = Field(
        description="The documents used to generate the answer",
        serialization_alias="contextDocuments",
    )
