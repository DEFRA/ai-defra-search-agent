import logging

import boto3
import fastapi
import pymongo.asynchronous.database

from app import config, dependencies
from app.bedrock import service as bedrock_service
from app.chat import agent, repository, service
from app.common import knowledge, mongo
from app.models import dependencies as model_dependencies
from app.models import service as model_service
from app.prompts.repository import FileSystemPromptRepository

logger = logging.getLogger(__name__)


def get_knowledge_retriever(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> knowledge.KnowledgeRetriever | None:
    return knowledge.KnowledgeRetriever(
        base_url=app_config.knowledge.base_url,
        similarity_threshold=app_config.knowledge.similarity_threshold,
    )


def get_prompt_repository() -> FileSystemPromptRepository:
    return FileSystemPromptRepository()


def get_bedrock_runtime_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> boto3.client:
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region,
        )

    return boto3.client("bedrock-runtime", region_name=app_config.aws_region)


def get_bedrock_client(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> boto3.client:
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region,
        )

    return boto3.client("bedrock", region_name=app_config.aws_region)


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
) -> repository.AbstractConversationRepository:
    return repository.MongoConversationRepository(db=db)


def get_chat_service(
    chat_agent: agent.AbstractChatAgent = fastapi.Depends(get_chat_agent),
    conversation_repository: repository.AbstractConversationRepository = fastapi.Depends(
        get_conversation_repository
    ),
    model_resolution_service: model_service.AbstractModelResolutionService = fastapi.Depends(
        model_dependencies.get_model_resolution_service
    ),
) -> service.ChatService:
    return service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
        model_resolution_service=model_resolution_service,
    )
