import datetime
import uuid
from typing import Literal

import pydantic
import pydantic.alias_generators


class BaseRequestSchema(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class FeedbackRequest(BaseRequestSchema):
    conversation_id: uuid.UUID | None = pydantic.Field(
        default=None,
        description="The ID of the conversation this feedback relates to",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    was_helpful: Literal[
        "very-useful", "useful", "neither", "not-useful", "not-at-all-useful"
    ] = pydantic.Field(
        description="User's rating of how useful the AI Assistant was",
        examples=["very-useful"],
    )
    comment: str | None = pydantic.Field(
        default=None,
        description="Optional user comments provided by the user about the AI assistant's response",
        max_length=1200,
        examples=["The response was very informative and addressed my question well."],
    )


class FeedbackResponse(BaseRequestSchema):
    feedback_id: uuid.UUID = pydantic.Field(
        description="The ID of the submitted feedback", serialization_alias="feedbackId"
    )
    timestamp: datetime.datetime = pydantic.Field(
        description="The timestamp when the feedback was submitted",
    )
