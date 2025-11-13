import logging

import fastapi

from app.chat import api_schemas, dependencies, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post("/chat", response_model=api_schemas.ChatResponse)
async def chat(request: api_schemas.ChatRequest, chat_service: service.ChatService=fastapi.Depends(dependencies.get_chat_service)):
    
