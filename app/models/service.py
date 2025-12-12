import abc

from app import config
from app.models import models


class AbstractModelResolutionService(abc.ABC):
    @abc.abstractmethod
    def get_available_models(self) -> list[models.ModelInfo]:
        """Retrieve a list of available models."""

    @abc.abstractmethod
    def resolve_model(self, model_id: str) -> models.ModelInfo:
        """Resolve a model by its ID."""


class ConfigModelResolutionService(AbstractModelResolutionService):
    def __init__(self, app_config: config.AppConfig):
        self.app_config = app_config

    def get_available_models(self) -> list[models.ModelInfo]:
        # TODO: Project multiple guardrails as separate available models

        available_models = self.app_config.bedrock.available_generation_models.values()

        return [
            models.ModelInfo(
                name=model.name,
                description=model.description,
                model_id=model.model_id,
            )
            for model in available_models
        ]

    def resolve_model(self, model_id: str) -> models.ModelInfo:
        """Resolve a model by its internal ID."""
        available_models = self.app_config.bedrock.available_generation_models

        if model_id not in available_models:
            msg = f"Model '{model_id}' not found"
            raise ValueError(msg)

        model = available_models[model_id]
        return models.ModelInfo(
            name=model.name,
            description=model.description,
            model_id=model.model_id,
        )
