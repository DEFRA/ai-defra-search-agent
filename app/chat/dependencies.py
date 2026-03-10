import logging

import boto3
import botocore.config
import fastapi
import pymongo.asynchronous.database

from app import config, dependencies
from app.bedrock import service as bedrock_service
from app.chat import agent, repository, service
from app.common import knowledge, mongo, sqs
from app.models import dependencies as model_dependencies
from app.models import service as model_service
from app.prompts.repository import FileSystemPromptRepository

logger = logging.getLogger(__name__)


def get_knowledge_retriever(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> knowledge.KnowledgeRetriever | None:
    return knowledge.KnowledgeRetriever(
        base_url=app_config.knowledge.base_url,
    )


def get_prompt_repository() -> FileSystemPromptRepository:
    return FileSystemPromptRepository()


def _bedrock_client_kwargs(app_config: config.AppConfig) -> dict:
    kwargs: dict = {
        "region_name": app_config.sqs.region,
        "config": botocore.config.Config(
            connect_timeout=app_config.bedrock.connect_timeout,
            read_timeout=app_config.bedrock.read_timeout,
        ),
    }
    if app_config.bedrock.use_credentials:
        kwargs["aws_access_key_id"] = app_config.bedrock.access_key_id
        kwargs["aws_secret_access_key"] = app_config.bedrock.secret_access_key
    if app_config.bedrock.endpoint_url:
        kwargs["endpoint_url"] = app_config.bedrock.endpoint_url
    return kwargs


def get_bedrock_runtime_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> boto3.client:
    return boto3.client("bedrock-runtime", **_bedrock_client_kwargs(app_config))


def get_bedrock_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> boto3.client:
    return boto3.client("bedrock", **_bedrock_client_kwargs(app_config))


def get_bedrock_inference_service(
    api_client: boto3.client = fastapi.Depends(get_bedrock_client),
    runtime_client: boto3.client = fastapi.Depends(get_bedrock_runtime_client),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
    knowledge_retriever: knowledge.KnowledgeRetriever | None = fastapi.Depends(
        get_knowledge_retriever
    ),
) -> bedrock_service.BedrockInferenceService:
    return bedrock_service.BedrockInferenceService(
        api_client=api_client,
        runtime_client=runtime_client,
        app_config=app_config,
        knowledge_retriever=knowledge_retriever,
    )


def get_chat_agent(
    inference_service: bedrock_service.BedrockInferenceService = fastapi.Depends(
        get_bedrock_inference_service
    ),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
    prompt_repository: FileSystemPromptRepository = fastapi.Depends(
        get_prompt_repository
    ),
) -> agent.AbstractChatAgent:
    return agent.BedrockChatAgent(
        inference_service=inference_service,
        app_config=app_config,
        prompt_repository=prompt_repository,
    )


def get_conversation_repository(
    db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> repository.AbstractConversationRepository:
    return repository.MongoConversationRepository(
        db=db,
        retry_attempts=app_config.mongo.retry_attempts,
        retry_base_delay_seconds=app_config.mongo.retry_base_delay_seconds,
    )


def get_sqs_client() -> sqs.SQSClient:
    return sqs.SQSClient()


def get_chat_service(
    chat_agent: agent.AbstractChatAgent = fastapi.Depends(get_chat_agent),
    conversation_repository: repository.AbstractConversationRepository = fastapi.Depends(
        get_conversation_repository
    ),
    model_resolution_service: model_service.AbstractModelResolutionService = fastapi.Depends(
        model_dependencies.get_model_resolution_service
    ),
    sqs_client: sqs.SQSClient = fastapi.Depends(get_sqs_client),
) -> service.ChatService:
    return service.ChatService(
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
        chat_agent=chat_agent,
    )


def get_queue_chat_service(
    conversation_repository: repository.AbstractConversationRepository = fastapi.Depends(
        get_conversation_repository
    ),
    model_resolution_service: model_service.AbstractModelResolutionService = fastapi.Depends(
        model_dependencies.get_model_resolution_service
    ),
    sqs_client: sqs.SQSClient = fastapi.Depends(get_sqs_client),
) -> service.ChatService:
    """Lightweight ChatService for queue-only operations. Skips Bedrock/agent deps for fast response."""
    return service.ChatService(
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
        sqs_client=sqs_client,
    )


def get_model_resolution_service() -> model_service.AbstractModelResolutionService:
    return model_service.ConfigModelResolutionService(config.config)


async def initialize_worker_services():
    """Initialize services for worker without FastAPI dependency injection context."""
    app_config = config.get_config()

    mongo_client = await mongo.get_mongo_client(app_config)
    db = mongo_client[app_config.mongo.database]

    runtime_kwargs = _bedrock_client_kwargs(app_config)
    api_kwargs = _bedrock_client_kwargs(app_config)
    bedrock_runtime = boto3.client("bedrock-runtime", **runtime_kwargs)
    bedrock_client = boto3.client("bedrock", **api_kwargs)

    knowledge_retriever = knowledge.KnowledgeRetriever(
        base_url=app_config.knowledge.base_url,
    )

    inference_service = bedrock_service.BedrockInferenceService(
        api_client=bedrock_client,
        runtime_client=bedrock_runtime,
        app_config=app_config,
        knowledge_retriever=knowledge_retriever,
    )

    prompt_repo = FileSystemPromptRepository()

    chat_agent = agent.BedrockChatAgent(
        inference_service=inference_service,
        app_config=app_config,
        prompt_repository=prompt_repo,
    )

    conversation_repo = repository.MongoConversationRepository(
        db=db,
        retry_attempts=app_config.mongo.retry_attempts,
        retry_base_delay_seconds=app_config.mongo.retry_base_delay_seconds,
    )

    model_resolution = model_service.ConfigModelResolutionService(app_config)

    sqs_client = sqs.SQSClient()

    chat_svc = service.ChatService(
        conversation_repository=conversation_repo,
        model_resolution_service=model_resolution,
        sqs_client=sqs_client,
        chat_agent=chat_agent,
    )

    return chat_svc, conversation_repo, sqs_client
