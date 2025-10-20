from pydantic import BaseModel, Field

class MessageResponse(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., user, assistant)")
    content: str = Field(..., description="Content of the message")
    

class TokenUsageResponse(BaseModel):
    input_tokens: int = Field(..., description="Number of input tokens used", serialization_alias="inputTokens")
    output_tokens: int = Field(..., description="Number of output tokens used", serialization_alias="outputTokens")
    total_tokens: int = Field(..., description="Total number of tokens used", serialization_alias="totalTokens")
    model: str = Field(..., description="Name of the LLM model used")
    stage_name: str = Field(..., description="Name of the processing stage", serialization_alias="stageName")


class ConversationHistoryResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation", serialization_alias="conversationId")
    messages: list[MessageResponse] = Field(..., description="List of messages in the conversation history")
    token_usage: list[TokenUsageResponse] = Field(..., description="List of token usage records", serialization_alias="tokenUsage")
