from app.models import service


def get_model_resolution_service() -> service.AbstractModelResolutionService:
    return service.ConfigModelResolutionService()
