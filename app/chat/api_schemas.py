import datetime
import uuid

import pydantic
import pydantic.alias_generators


class BaseRequestSchema(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class ChatRequest(BaseRequestSchema):
    question: str = pydantic.Field(
        description="The user's question to ask the chat agent",
        min_length=1,
        examples=["What is the weather like today?"],
    )
    conversation_id: uuid.UUID | None = pydantic.Field(
        default=None,
        description="The ID of an existing conversation to continue. If not provided, a new conversation will be started.",
        examples=["3fa85f64-5717-4562-b374-2c963f66afa6"],
    )
    model_id: str = pydantic.Field(
        description="The internal id of the model to use for generating the response",
        examples=["anthropic.claude-3-7-sonnet"],
    )


class MessageResponse(pydantic.BaseModel):
    message_id: uuid.UUID = pydantic.Field(
        description="The unique identifier for the message",
        serialization_alias="messageId",
    )
    role: str = pydantic.Field(
        description="The role of the message sender, e.g., 'user' or 'assistant'"
    )
    content: str = pydantic.Field(description="The content of the message")
    model_name: str | None = pydantic.Field(
        default=None,
        description="The name of the model used to generate the message",
        exclude_if=lambda v: v is None,
        serialization_alias="modelName",
    )
    model_id: str | None = pydantic.Field(
        default=None,
        description="The internal model id of the model used to generate the message, if applicable",
        exclude_if=lambda v: v is None,
        serialization_alias="modelId",
    )
    status: str = pydantic.Field(description="The status of the message processing")
    error_message: str | None = pydantic.Field(
        default=None,
        description="Error message if the message processing failed",
        exclude_if=lambda v: v is None,
        serialization_alias="errorMessage",
    )
    timestamp: datetime.datetime = pydantic.Field(
        description="The timestamp of when the message was created"
    )


class QueueChatResponse(pydantic.BaseModel):
    message_id: uuid.UUID = pydantic.Field(
        description="The unique identifier for the queued message",
        serialization_alias="messageId",
    )
    conversation_id: uuid.UUID = pydantic.Field(
        description="The ID of the conversation", serialization_alias="conversationId"
    )
    status: str = pydantic.Field(
        description="The current status of the message processing"
    )


class ChatResponse(pydantic.BaseModel):
    conversation_id: uuid.UUID = pydantic.Field(
        description="The ID of the conversation", serialization_alias="conversationId"
    )
    messages: list[MessageResponse] = pydantic.Field(
        description="The list of messages in the conversation",
    )
