from logging import getLogger

from fastapi import APIRouter
from fastapi.params import Depends

from app.conversation_history.dependencies import get_conversation_history_service
from app.conversation_history.service import ConversationHistoryService
from app.v2_chat.agent import LangGraphChatAgent
from app.v2_chat.api_schemas import ChatRequest, ChatResponse, ContextDocumentResponse
from app.v2_chat.service import ChatService

logger = getLogger(__name__)

router = APIRouter(tags=["chat"])


def get_chat_service(history_service: ConversationHistoryService = Depends(get_conversation_history_service)):
    orchestrator = LangGraphChatAgent()
    return ChatService(orchestrator, history_service)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chat_service: ChatService=Depends(get_chat_service)):
    response, conversation_id = await chat_service.execute_chat(request.question)

    context_documents = [
        ContextDocumentResponse(
            content=doc.content,
            title=doc.metadata.get("title", "Unknown Title"),
            snapshot_id=doc.snapshot_id,
            source_id=doc.source_id
        )
        for doc in response.get("context", [])
    ]

    return ChatResponse(
        answer=response.get("answer", ""),
        conversation_id=str(conversation_id),
        context_documents=context_documents,
    )
