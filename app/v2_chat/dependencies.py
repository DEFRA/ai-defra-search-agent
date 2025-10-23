import fastapi
import pymongo.asynchronous.database

from app.common import mongo
from app.conversation_history import repository
from app.conversation_history import service as conv_service
from app.v2_chat import agent, service


def get_chat_agent() -> agent.AbstractChatAgent:
    """Factory for creating chat agent instances."""
    return agent.LangGraphChatAgent()


def get_conversation_history_service(
    db: pymongo.asynchronous.database.AsyncDatabase = fastapi.Depends(mongo.get_db)
) -> conv_service.ConversationHistoryService:
    """Factory for creating conversation history service instances."""
    repo = repository.MongoConversationHistoryRepository(db)
    return conv_service.ConversationHistoryService(repo)


def get_chat_service(
    orchestrator: agent.AbstractChatAgent = fastapi.Depends(get_chat_agent),
    history_service: conv_service.ConversationHistoryService = fastapi.Depends(get_conversation_history_service)
) -> service.ChatService:
    """Factory for creating chat service instances with all required dependencies."""
    return service.ChatService(orchestrator, history_service)
