from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class BedrockConfig(BaseSettings):
    model_config = SettingsConfigDict()
    use_credentials: bool = Field(default=False, alias="AWS_BEDROCK_USE_CREDENTIALS")
    access_key_id: str | None = Field(default=None, alias="AWS_BEDROCK_ACCESS_KEY_ID")
    secret_access_key: str | None = Field(default=None, alias="AWS_BEDROCK_SECRET_ACCESS_KEY")
    guardrail_identifier: str | None = Field(default=None, alias="AWS_BEDROCK_GUARDRAIL_IDENTIFIER")
    guardrail_version: str | None = Field(default=None, alias="AWS_BEDROCK_GUARDRAIL_VERSION")
    generation_model: str = Field(..., alias="AWS_BEDROCK_GENERATION_MODEL")
    grading_model: str = Field(default="default-grading-model", alias="AWS_BEDROCK_MODEL_GRADING")
    provider: str = Field(default="anthropic", alias="AWS_BEDROCK_PROVIDER")
    embedding_model: str = Field(..., alias="AWS_BEDROCK_EMBEDDING_MODEL")


class MongoConfig(BaseSettings):
    model_config = SettingsConfigDict()
    uri: str
    database: str = "ai-defra-search-agent"
    truststore: str = "TRUSTSTORE_CDP_ROOT_CA"

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict()
    python_env: str = "production"
    host: str
    port: int
    log_config: str
    aws_region: str
    localstack_url: str | None = None
    http_proxy: HttpUrl | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"
    prompt_s3_bucket: str

    mongo: MongoConfig = MongoConfig()
    bedrock: BedrockConfig = BedrockConfig()


config = AppConfig()
