import pydantic


class MessageResponse(pydantic.BaseModel):
    role: str = pydantic.Field(..., description="Role of the message sender (e.g., user, assistant)")
    content: str = pydantic.Field(..., description="Content of the message")
    sources: list[dict[str, str]] | None = pydantic.Field(
        None,
        description="List of source documents related to the message",
        serialization_alias="sources"
    )


class TokenUsageResponse(pydantic.BaseModel):
    input_tokens: int = pydantic.Field(..., description="Number of input tokens used", serialization_alias="inputTokens")
    output_tokens: int = pydantic.Field(..., description="Number of output tokens used", serialization_alias="outputTokens")
    total_tokens: int = pydantic.Field(..., description="Total number of tokens used", serialization_alias="totalTokens")
    model: str = pydantic.Field(..., description="Name of the LLM model used")
    stage_name: str = pydantic.Field(..., description="Name of the processing stage", serialization_alias="stageName")


class ConversationHistoryResponse(pydantic.BaseModel):
    conversation_id: str = pydantic.Field(..., description="Unique identifier for the conversation", serialization_alias="conversationId")
    messages: list[MessageResponse] = pydantic.Field(..., description="List of messages in the conversation history")
    token_usage: list[TokenUsageResponse] = pydantic.Field(..., description="List of token usage records", serialization_alias="tokenUsage")
