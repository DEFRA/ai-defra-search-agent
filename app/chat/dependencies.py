import fastapi

from app.chat import agent, service


def get_chat_agent() -> agent.AbstractChatAgent:
    pass


def get_chat_service(
    chat_agent: agent.AbstractChatAgent = fastapi.Depends(get_chat_agent),
) -> service.ChatService:
    return service.ChatService(chat_agent=chat_agent)
