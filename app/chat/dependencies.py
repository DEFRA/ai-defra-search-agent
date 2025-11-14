import fastapi

from app.chat import agent, service, repository


def get_chat_agent() -> agent.AbstractChatAgent:
    pass


def get_conversation_repository() -> repository.AbstractConversationRepository:
    pass


def get_chat_service(
    chat_agent: agent.AbstractChatAgent = fastapi.Depends(get_chat_agent),
    conversation_repository: repository.AbstractConversationRepository = fastapi.Depends(
        get_conversation_repository
    ),
) -> service.ChatService:
    return service.ChatService(
        chat_agent=chat_agent, conversation_repository=conversation_repository
    )
