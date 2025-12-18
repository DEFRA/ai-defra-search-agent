import dataclasses


class UnsupportedModelError(Exception):
    """Raised when a requested model is not found or supported."""


@dataclasses.dataclass
class ModelInfo:
    name: str
    description: str
    model_id: str
