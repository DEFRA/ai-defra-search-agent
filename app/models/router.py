import logging
from typing import Annotated

import fastapi

from app.models import api_schemas, dependencies, service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=list[api_schemas.ModelInfoResponse],
    summary="List available models",
    description="Retrieves a list of all AI models available for chat interactions.",
    responses={204: {"description": "No available models found"}},
)
async def list_models(
    model_resolution_service: Annotated[
        service.AbstractModelResolutionService,
        fastapi.Depends(dependencies.get_model_resolution_service),
    ],
):
    models = model_resolution_service.get_available_models()

    if len(models) == 0:
        raise fastapi.HTTPException(
            status_code=204, detail="No available models found."
        )

    return [
        api_schemas.ModelInfoResponse(
            model_id=model.model_id,
            model_name=model.name,
            model_description=model.description,
        )
        for model in models
    ]
