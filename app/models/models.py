import dataclasses


@dataclasses.dataclass
class ModelInfo:
    name: str
    description: str
    model_id: str
