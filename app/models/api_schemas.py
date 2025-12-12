import pydantic


class ModelInfoResponse(pydantic.BaseModel):
    model_id: str = pydantic.Field(
        description="The internal id of the model",
        examples=["anthropic.claude-3-7-sonnet"],
        serialization_alias="modelId",
    )
    model_name: str = pydantic.Field(
        description="The name of the model",
        examples=["Claude 3 Haiku"],
        serialization_alias="modelName",
    )
    model_description: str = pydantic.Field(
        description="A brief description of the model",
        examples=[
            "A helpful and creative AI model optimized for generating poetic responses."
        ],
        serialization_alias="modelDescription",
    )
