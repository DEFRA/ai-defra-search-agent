import fastapi

from app import config, dependencies
from app.models import service


def get_model_resolution_service(
    app_config: config.AppConfig = fastapi.Depends(dependencies.get_app_config),
) -> service.AbstractModelResolutionService:
    return service.ConfigModelResolutionService(app_config)
