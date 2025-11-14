import fastapi
import pymongo.asynchronous.database

from app.bedrock import service as bedrock_service
from app.chat import agent, repository, service
from app.common import bedrock as bedrock_clients
from app.common import mongo


def get_bedrock_inference_service() -> bedrock_service.BedrockInferenceService:
    return bedrock_service.BedrockInferenceService(
        api_client=bedrock_clients.get_bedrock_client(),
        runtime_client=bedrock_clients.get_bedrock_runtime_client(),
    )

def get_chat_agent() -> agent.AbstractChatAgent:
    return agent.BedrockChatAgent(
        inference_service=get_bedrock_inference_service()
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
        chat_agent=chat_agent, conversation_repository=conversation_repository
    )
