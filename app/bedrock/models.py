import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    id: str
    guardrail_id: str | None = None
    guardrail_version: str | None = None


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    model_id: str
    content: list[dict[str, Any]]
    usage: dict[str, int]


@dataclasses.dataclass(frozen=True)
class InferenceProfile:
    id: str
    name: str
    models: list[dict[str, Any]]
