from logging import getLogger

from fastapi import APIRouter

from app.v2_chat.api_schemas import ChatRequest, ChatResponse
from app.v2_chat.agent import LangGraphChatAgent
from app.v2_chat.service import ChatService
from fastapi.params import Depends

logger = getLogger(__name__)

router = APIRouter(tags=["chat"])


def get_chat_service():
    orchestrator = LangGraphChatAgent()
    service = ChatService(orchestrator)
    return service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chat_service=Depends(get_chat_service)):
    logger.info(f"Received chat request: {request}")
    response = await chat_service.execute_chat(request.question)

    return {"answer": response}
