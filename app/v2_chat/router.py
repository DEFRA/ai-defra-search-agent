import logging

import fastapi
import fastapi.params

from app.conversation_history import dependencies as conv_deps
from app.conversation_history import service as conv_service
from app.v2_chat import agent, api_schemas, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


def get_chat_service(history_service: conv_service.ConversationHistoryService = fastapi.params.Depends(conv_deps.get_conversation_history_service)):
    orchestrator = agent.LangGraphChatAgent()
    return service.ChatService(orchestrator, history_service)


@router.post("/chat", response_model=api_schemas.ChatResponse)
async def chat(request: api_schemas.ChatRequest, chat_service: service.ChatService=fastapi.params.Depends(get_chat_service)):
    response, conversation_id = await chat_service.execute_chat(
        request.question,
        request.conversation_id
    )

    context_documents = [
        api_schemas.ContextDocumentResponse(
            content=doc.content,
            title=doc.metadata.get("title", "Unknown Title"),
            snapshot_id=doc.snapshot_id,
            source_id=doc.source_id
        )
        for doc in response.get("context", [])
    ]

    usage = [
        api_schemas.TokenUsageResponse(
            model=token_usage.model,
            stage_name=token_usage.stage_name,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens
        )
        for token_usage in response.get("token_usage", [])
    ]

    return api_schemas.ChatResponse(
        answer=response.get("answer", ""),
        conversation_id=str(conversation_id),
        context_documents=context_documents,
        usage=usage
    )
