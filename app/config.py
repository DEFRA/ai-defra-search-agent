import logging

import pydantic
import pydantic_settings

logger = logging.getLogger(__name__)


class BedrockConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict()
    use_credentials: bool = pydantic.Field(default=False, alias="AWS_BEDROCK_USE_CREDENTIALS")
    access_key_id: str | None = pydantic.Field(default=None, alias="AWS_BEDROCK_ACCESS_KEY_ID")
    secret_access_key: str | None = pydantic.Field(default=None, alias="AWS_BEDROCK_SECRET_ACCESS_KEY")
    guardrail_identifier: str | None = pydantic.Field(default=None, alias="AWS_BEDROCK_GUARDRAIL_IDENTIFIER")
    guardrail_version: str | None = pydantic.Field(default=None, alias="AWS_BEDROCK_GUARDRAIL_VERSION")
    generation_model: str = pydantic.Field(..., alias="AWS_BEDROCK_GENERATION_MODEL")
    grading_model: str = pydantic.Field(default="default-grading-model", alias="AWS_BEDROCK_MODEL_GRADING")
    provider: str = pydantic.Field(default="anthropic", alias="AWS_BEDROCK_PROVIDER")
    embedding_model: str = pydantic.Field(..., alias="AWS_BEDROCK_EMBEDDING_MODEL")


class MongoConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict()
    uri: str = pydantic.Field(..., alias="MONGO_URI")
    database: str = pydantic.Field(default="ai-defra-search-agent", alias="MONGO_DATABASE")
    truststore: str = pydantic.Field(default="TRUSTSTORE_CDP_ROOT_CA", alias="MONGO_TRUSTSTORE")


class ChatWorkflowConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict()
    data_service_url: pydantic.HttpUrl = pydantic.Field(..., alias="DATA_SERVICE_URL")
    default_knowledge_group_id: str = pydantic.Field(..., alias="DEFAULT_KNOWLEDGE_GROUP_ID")


class AppConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict()
    python_env: str = "production"
    host: str | None = None
    port: int
    log_config: str
    aws_region: str
    localstack_url: str | None = None
    http_proxy: str | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"

    mongo: MongoConfig = pydantic.Field(default_factory=MongoConfig)
    bedrock: BedrockConfig = pydantic.Field(default_factory=BedrockConfig)
    workflow: ChatWorkflowConfig = pydantic.Field(default_factory=ChatWorkflowConfig)


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
                    "url": error["url"]
                }
                for error in e.errors()
            ]

            logger.error("Config validation failed with errors: %s", error_details)

            msg = "Invalid application configuration"
            raise RuntimeError(msg) from None

    return config
