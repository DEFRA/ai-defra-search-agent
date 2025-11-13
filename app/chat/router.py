import logging

import fastapi

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request, chat_service):
    pass
