import logging

import boto3
import fastapi
import pymongo.asynchronous.database

from app import config, dependencies
from app.bedrock import service as bedrock_service
from app.chat import agent, repository, service
from app.common import mongo

logger = logging.getLogger(__name__)


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
) -> bedrock_service.BedrockInferenceService:
    return bedrock_service.BedrockInferenceService(
        api_client=api_client, runtime_client=runtime_client, app_config=app_config
    )


def get_chat_agent(
    inference_service: bedrock_service.BedrockInferenceService = fastapi.Depends(
        get_bedrock_inference_service
    ),
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> agent.AbstractChatAgent:
    return agent.BedrockChatAgent(
        inference_service=inference_service, app_config=app_config
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
) -> service.ChatService:
    return service.ChatService(
        chat_agent=chat_agent,
        conversation_repository=conversation_repository,
    )
