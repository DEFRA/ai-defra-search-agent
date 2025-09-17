from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict()
    python_env: str | None = None
    host: str | None = None
    port: int | None = None
    log_config: str | None = None
    mongo_uri: str | None = None
    mongo_database: str = "ai-defra-search-agent"
    mongo_truststore: str = "TRUSTSTORE_CDP_ROOT_CA"
    aws_endpoint_url: str | None = None
    http_proxy: HttpUrl | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"

    # AWS
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str | None = None

    # Bedrock
    AWS_ACCESS_KEY_ID_BEDROCK: str | None = None
    AWS_SECRET_ACCESS_KEY_BEDROCK: str | None = None
    AWS_REGION_BEDROCK: str | None = None
    AWS_BEDROCK_MODEL: str | None = None
    AWS_USE_CREDENTIALS_BEDROCK: str | None = None
    AWS_BEDROCK_GUARDRAIL: str | None = None
    AWS_BEDROCK_GUARDRAIL_VERSION: str | None = None

    # Anthropic
    ANTHROPIC_MAX_TOKENS: int | None = None
    ANTHROPIC_TEMPERATURE: float | None = None


config = AppConfig()
