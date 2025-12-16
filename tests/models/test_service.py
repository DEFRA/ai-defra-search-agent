import pytest

from app import config
from app.models import service


def test_resolve_model_raises_value_error_if_not_found(mocker):
    mock_app_config = mocker.Mock(spec=config.AppConfig)
    mock_app_config.bedrock = mocker.Mock()
    mock_app_config.bedrock.available_generation_models = {}

    resolution_service = service.ConfigModelResolutionService(
        app_config=mock_app_config
    )

    with pytest.raises(ValueError, match="Model 'unknown-model' not found"):
        resolution_service.resolve_model("unknown-model")
