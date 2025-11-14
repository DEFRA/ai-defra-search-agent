import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    model: str
    content: list[dict[str, any]]


@dataclasses.dataclass(frozen=True)
class InferenceProfile:
    id: str
    name: str
    models: list[str]
