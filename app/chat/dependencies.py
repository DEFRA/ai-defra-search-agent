import logging
from functools import lru_cache

import boto3
import fastapi
import pymongo.asynchronous.database

from app import config
from app.bedrock import service as bedrock_service
from app.chat import agent, repository, service
from app.common import mongo

logger = logging.getLogger(__name__)
app_config = config.get_config()


@lru_cache(maxsize=1)
def get_bedrock_runtime_client() -> boto3.client:
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region,
        )

    return boto3.client("bedrock-runtime", region_name=app_config.aws_region)


@lru_cache(maxsize=1)
def get_bedrock_client() -> boto3.client:
    if app_config.bedrock.use_credentials:
        return boto3.client(
            "bedrock",
            aws_access_key_id=app_config.bedrock.access_key_id,
            aws_secret_access_key=app_config.bedrock.secret_access_key,
            region_name=app_config.aws_region,
        )

    return boto3.client("bedrock", region_name=app_config.aws_region)


def get_bedrock_inference_service() -> bedrock_service.BedrockInferenceService:
    return bedrock_service.BedrockInferenceService(
        api_client=get_bedrock_client(),
        runtime_client=get_bedrock_runtime_client(),
    )


def get_chat_agent(
    inference_service: bedrock_service.BedrockInferenceService = fastapi.Depends(get_bedrock_inference_service)
) -> agent.AbstractChatAgent:
    return agent.BedrockChatAgent(
        inference_service=inference_service
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
