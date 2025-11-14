import uuid

import pydantic


class ChatRequest(pydantic.BaseModel):
    question: str = pydantic.Field(
        description="The user's question to ask the chat agent",
        min_length=1,
        examples=["What is the weather like today?"],
    )
    conversation_id: uuid.UUID | None = pydantic.Field(
        default=None,
        description="The ID of an existing conversation to continue. If not provided, a new conversation will be started.",
        examples=["3fa85f64-5717-4562-b374-2c963f66afa6"],
        alias="conversationId"
    )


class MessageResponse(pydantic.BaseModel):
    role: str = pydantic.Field(description="The role of the message sender, e.g., 'user' or 'assistant'")
    content: str = pydantic.Field(description="The content of the message")
    model: str | None = pydantic.Field(
        default=None,
        description="The model used to generate the message, if applicable",
        exclude_if=lambda v: v is None,
    )


class ChatResponse(pydantic.BaseModel):
    conversation_id: uuid.UUID = pydantic.Field(
        description="The ID of the conversation",
        serialization_alias="conversationId"
    )
    messages: list[MessageResponse] = pydantic.Field(
        description="The list of messages in the conversation",
        min_length=2,
    )
