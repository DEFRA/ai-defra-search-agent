import json
import logging
from typing import Annotated

import pydantic
import pydantic_settings

logger = logging.getLogger(__name__)


class BedrockGuardrailConfig(pydantic.BaseModel):
    guardrail_id: str
    guardrail_version: str = pydantic.Field(..., pattern=r"(|([1-9][0-9]{0,7})|(DRAFT))")


class BedrockModelConfig(pydantic.BaseModel):
    name: str
    description: str
    id: str
    guardrails: BedrockGuardrailConfig | None = None

    def __hash__(self):
        return hash(self.name)


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
                if model_info["name"] in models:
                    msg = f"Duplicate model name found in configuration: {model_info['name']}"
                    raise ValueError(msg)

                guardrails = None

                if "guardrails" in model_info:
                    guardrails = BedrockGuardrailConfig(
                        guardrail_id=model_info["guardrails"]["guardrail_id"],
                        guardrail_version=model_info["guardrails"]["guardrail_version"],
                    )

                models[model_info["name"]] = BedrockModelConfig(
                    name=model_info["name"],
                    id=model_info["id"],
                    description=model_info["description"],
                    guardrails=guardrails,
                )

            return models

        return v


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
    host: str | None = None
    port: int = pydantic.Field(...)
    log_config: str = pydantic.Field(...)
    aws_region: str = pydantic.Field(...)
    localstack_url: str | None = None
    http_proxy: str | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"

    mongo: MongoConfig = pydantic.Field(default_factory=MongoConfig)
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

            logger.error("Config validation failed with errors: %s", error_details)

            msg = "Invalid application configuration"
            raise RuntimeError(msg) from None

    return config
