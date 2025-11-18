import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    id: str
    guardrail_id: str | None
    guardrail_version: str | None


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    model: str
    content: list[dict[str, any]]


@dataclasses.dataclass(frozen=True)
class InferenceProfile:
    id: str
    name: str
    models: list[str]
