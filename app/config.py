import json
import logging
from typing import Annotated

import pydantic
import pydantic_settings

logger = logging.getLogger(__name__)


class BedrockGuardrailConfig(pydantic.BaseModel):
    guardrail_id: str = pydantic.Field(
        ...,
        pattern=r"^arn:aws:bedrock:[a-z]{2}-[a-z]+-\d{1}:\d{12}:guardrail/[a-z0-9]+$",
    )
    guardrail_version: str = pydantic.Field(..., pattern=r"^(([1-9][0-9]{0,7})|DRAFT)$")


class BedrockModelConfig(pydantic.BaseModel):
    name: str
    description: str
    bedrock_model_id: str
    model_id: str
    guardrails: BedrockGuardrailConfig | None = None


class BedrockConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", extra="ignore")
    use_credentials: bool = pydantic.Field(
        default=False, alias="AWS_BEDROCK_USE_CREDENTIALS"
    )
    access_key_id: str | None = pydantic.Field(
        default=None, alias="AWS_BEDROCK_ACCESS_KEY_ID"
    )
    secret_access_key: str | None = pydantic.Field(
        default=None, alias="AWS_BEDROCK_SECRET_ACCESS_KEY"
    )
    available_generation_models: Annotated[
        dict[str, BedrockModelConfig], pydantic_settings.NoDecode
    ] = pydantic.Field(
        ...,
        alias="AWS_BEDROCK_AVAILABLE_GENERATION_MODELS",
    )
    default_generation_model: str = pydantic.Field(
        ..., alias="AWS_BEDROCK_DEFAULT_GENERATION_MODEL"
    )
    max_response_tokens: int = pydantic.Field(
        default=1024, alias="AWS_BEDROCK_MAX_RESPONSE_TOKENS"
    )
    default_model_temprature: float = pydantic.Field(
        default=0.7, alias="AWS_BEDROCK_DEFAULT_MODEL_TEMPERATURE"
    )

    @pydantic.field_validator("available_generation_models", mode="before")
    @classmethod
    def parse_bedrock_model(cls, v):
        if isinstance(v, str):
            parsed = json.loads(v)

            models = {}

            for model_info in parsed:
                if model_info["modelId"] in models:
                    msg = f"Duplicate model id found in configuration: {model_info['modelId']}"
                    raise ValueError(msg)

                guardrails = None

                if "guardrails" in model_info:
                    guardrails = BedrockGuardrailConfig(
                        guardrail_id=model_info["guardrails"]["guardrail_id"],
                        guardrail_version=model_info["guardrails"]["guardrail_version"],
                    )

                models[model_info["modelId"]] = BedrockModelConfig(
                    name=model_info["name"],
                    bedrock_model_id=model_info["bedrockModelId"],
                    model_id=model_info["modelId"],
                    description=model_info["description"],
                    guardrails=guardrails,
                )

            return models

        return v


class KnowledgeConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", extra="ignore")
    base_url: str = pydantic.Field(..., alias="KNOWLEDGE_BASE_URL")
    knowledge_group_id: str = pydantic.Field(..., alias="KNOWLEDGE_GROUP_ID")
    similarity_threshold: float = pydantic.Field(
        default=0.5, alias="KNOWLEDGE_SIMILARITY_THRESHOLD"
    )


class MongoConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", extra="ignore")
    uri: str = pydantic.Field(..., alias="MONGO_URI")
    database: str = pydantic.Field(
        default="ai-defra-search-agent", alias="MONGO_DATABASE"
    )
    truststore: str = pydantic.Field(
        default="TRUSTSTORE_CDP_ROOT_CA", alias="MONGO_TRUSTSTORE"
    )


class AppConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", extra="ignore")
    python_env: str = "production"
    host: str = "127.0.0.1"
    port: int = 8086
    log_config: str = pydantic.Field(...)
    aws_region: str = pydantic.Field(...)
    localstack_url: str | None = None
    sqs_chat_queue_url: str = pydantic.Field(..., alias="SQS_CHAT_QUEUE_URL")
    http_proxy: str | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"

    mongo: MongoConfig = pydantic.Field(default_factory=MongoConfig)
    knowledge: KnowledgeConfig = pydantic.Field(default_factory=KnowledgeConfig)
    bedrock: BedrockConfig = pydantic.Field(default_factory=BedrockConfig)


config: AppConfig | None = None


def get_config() -> AppConfig:
    global config
    if config is None:
        try:
            config = AppConfig()
        except pydantic.ValidationError as e:
            error_details = [
                {
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "type": error["type"],
                    "message": error["msg"],
                    "url": error.get("url"),
                }
                for error in e.errors()
            ]

            error_strings = [
                f"Field '{error['field']}' {error['message']}"
                for error in error_details
            ]

            msg = f"Config validation failed with errors: {', '.join(error_strings)}"
            logger.error(msg)
            raise RuntimeError(msg) from None
    return config
