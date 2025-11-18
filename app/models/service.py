import abc

from app import config

from app.models import models

app_config = config.get_config()


class AbstractModelResolutionService(abc.ABC):
    @abc.abstractmethod
    def get_available_models(self) -> list[models.ModelInfo]:
        """Retrieve a list of available models."""
        pass


class ConfigModelResolutionService(AbstractModelResolutionService):
    def get_available_models(self) -> list[models.ModelInfo]:
        # TODO: Project multiple guardrails as separate available models

        available_models = app_config.bedrock.available_generation_models.values()

        return [
            models.ModelInfo(
                name=model.name,
                description=model.description,
                id=model.id,
            )
            for model in available_models
        ]
