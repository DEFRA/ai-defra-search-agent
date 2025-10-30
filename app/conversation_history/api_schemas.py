import pydantic


class MessageResponse(pydantic.BaseModel):
    role: str = pydantic.Field(
        ..., description="Role of the message sender (e.g., user, assistant)"
    )
    content: str = pydantic.Field(..., description="Content of the message")


class TokenUsageResponse(pydantic.BaseModel):
    input_tokens: int = pydantic.Field(
        ...,
        description="Number of input tokens used",
        serialization_alias="inputTokens",
    )
    output_tokens: int = pydantic.Field(
        ...,
        description="Number of output tokens used",
        serialization_alias="outputTokens",
    )
    total_tokens: int = pydantic.Field(
        ...,
        description="Total number of tokens used",
        serialization_alias="totalTokens",
    )
    model: str = pydantic.Field(..., description="Name of the LLM model used")
    stage_name: str = pydantic.Field(
        ..., description="Name of the processing stage", serialization_alias="stageName"
    )


class ConversationHistoryResponse(pydantic.BaseModel):
    conversation_id: str = pydantic.Field(
        ...,
        description="Unique identifier for the conversation",
        serialization_alias="conversationId",
    )
    messages: list[MessageResponse] = pydantic.Field(
        ..., description="List of messages in the conversation history"
    )
    token_usage: list[TokenUsageResponse] = pydantic.Field(
        ..., description="List of token usage records", serialization_alias="tokenUsage"
    )


class BulkTokenUsageRequest(pydantic.BaseModel):
    conversation_ids: list[str] = pydantic.Field(
        ...,
        description="List of conversation IDs to query",
        serialization_alias="conversationIds",
    )


class TokenUsageSummary(pydantic.BaseModel):
    total_input_tokens: int = pydantic.Field(
        ..., description="Total input tokens", serialization_alias="totalInputTokens"
    )
    total_output_tokens: int = pydantic.Field(
        ..., description="Total output tokens", serialization_alias="totalOutputTokens"
    )
    total_tokens: int = pydantic.Field(
        ...,
        description="Total tokens (input + output)",
        serialization_alias="totalTokens",
    )


class ModelTokenUsageSummary(TokenUsageSummary):
    model: str = pydantic.Field(..., description="Model name")


class ConversationTokenUsageSummary(TokenUsageSummary):
    conversation_id: str = pydantic.Field(
        ..., description="Conversation ID", serialization_alias="conversationId"
    )
    models: list[ModelTokenUsageSummary] = pydantic.Field(
        default_factory=list, description="Token usage by model for this conversation"
    )


class BulkTokenUsageResponse(pydantic.BaseModel):
    overall_usage: TokenUsageSummary = pydantic.Field(
        ...,
        description="Overall token usage across all conversations",
        serialization_alias="overallUsage",
    )
    usage_by_model: list[ModelTokenUsageSummary] = pydantic.Field(
        default_factory=list,
        description="Token usage aggregated by model",
        serialization_alias="usageByModel",
    )
    usage_by_conversation: list[ConversationTokenUsageSummary] = pydantic.Field(
        default_factory=list,
        description="Token usage by conversation",
        serialization_alias="usageByConversation",
    )
