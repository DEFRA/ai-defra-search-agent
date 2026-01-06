import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException
from pytest_mock import MockerFixture

from app.bedrock import models, service


@pytest.fixture
def bedrock_inference_service(
    mocker: MockerFixture, bedrock_client, bedrock_runtime_v2_client
):
    mock_config = mocker.Mock()
    mock_config.bedrock.max_response_tokens = 100
    mock_config.bedrock.default_model_temprature = 0.5

    return service.BedrockInferenceService(
        api_client=bedrock_client,
        runtime_client=bedrock_runtime_v2_client,
        app_config=mock_config,
    )


def test_no_guardrails_should_return_bedrock_response(
    bedrock_inference_service: service.BedrockInferenceService,
):
    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(id="geni-ai-3.5"),
        system_prompt="This is not a real prompt",
        messages=[
            {"role": "user", "content": [{"text": "What is the weather today?"}]}
        ],
    )

    assert response.model_id == "geni-ai-3.5"
    assert response.content == [{"text": "This is a stub response."}]


def test_invoke_with_inference_profile_should_return_model_id(
    bedrock_inference_service: service.BedrockInferenceService,
):
    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(
            id="arn:aws:bedrock:us-west-2:123456789012:inference-profile/geni-ai-3.5"
        ),
        system_prompt="This is not a real prompt",
        messages=[
            {"role": "user", "content": [{"text": "What is the weather today?"}]}
        ],
    )

    assert response.model_id == "geni-ai-3.5"
    assert response.content == [{"text": "This is a stub response."}]


def test_with_valid_guardrails_should_return_bedrock_response(
    bedrock_inference_service: service.BedrockInferenceService,
):
    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(
            id="geni-ai-3.5",
            guardrail_id="arn:aws:bedrock:us-west-2:123456789012:guardrail/8etdsfsdf3sd",
            guardrail_version=1,
        ),
        system_prompt="This is not a real prompt",
        messages=[
            {"role": "user", "content": [{"text": "What is the weather today?"}]}
        ],
    )

    assert response.model_id == "geni-ai-3.5"
    assert response.content == [{"text": "This is a stub response."}]


def test_guardrail_id_with_no_version_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(
        ValueError, match="The guardrail ID and version must be provided together"
    ):
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5",
                guardrail_id="arn:aws:bedrock:eu-central-1:123445511111:guardrail/8xqdsfsdf3gk",
                guardrail_version=None,
            ),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )


def test_guardrail_version_with_no_id_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(
        ValueError, match="The guardrail ID and version must be provided together"
    ):
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5", guardrail_id=None, guardrail_version=1
            ),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )


def test_missing_backing_model_should_raise_error(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    mocker.patch.object(
        bedrock_inference_service, "_get_backing_model", return_value=None
    )

    with pytest.raises(
        ValueError, match="Backing model not found for model ID: invalid-model-id"
    ):
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="invalid-model-id"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )


def test_get_inference_profile_details_with_valid_arn_should_return_profile(
    bedrock_inference_service: service.BedrockInferenceService,
):
    inference_profile = bedrock_inference_service.get_inference_profile_details(
        inference_profile_id="arn:aws:bedrock:us-west-2:123456789012:inference-profile/geni-ai-3.5"
    )

    assert (
        inference_profile.id
        == "arn:aws:bedrock:us-west-2:123456789012:inference-profile/geni-ai-3.5"
    )
    assert inference_profile.name == "Stub Inference Profile"
    assert inference_profile.models == [
        {"modelArn": "arn:aws:bedrock:eu-central-1::foundation-model/geni-ai-3.5"}
    ]


def test_get_inference_profile_details_with_model_id_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(ValueError, match="Invalid inference profile ID format"):
        bedrock_inference_service.get_inference_profile_details(
            inference_profile_id="geni-ai-3.5"
        )


def test_get_inference_profile_details_with_non_existent_profile_should_raise_error(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    mocker.patch.object(
        bedrock_inference_service.api_client, "get_inference_profile", return_value=None
    )

    with pytest.raises(ValueError, match="Inference profile not found"):
        bedrock_inference_service.get_inference_profile_details(
            inference_profile_id="arn:aws:bedrock:us-west-2:123456789012:inference-profile/non-existent-profile"
        )


def test_bedrock_400_error_should_raise_http_exception_400(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Invalid model parameters",
        },
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        side_effect=ClientError(error_response, "converse"),
    )

    with pytest.raises(HTTPException) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )

    assert exc_info.value.status_code == 400
    assert "Invalid request to AI model" in exc_info.value.detail
    assert "Invalid model parameters" in exc_info.value.detail


def test_bedrock_403_error_should_raise_http_exception_403(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    error_response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "User does not have access to model",
        },
        "ResponseMetadata": {"HTTPStatusCode": 403},
    }
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        side_effect=ClientError(error_response, "converse"),
    )

    with pytest.raises(HTTPException) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )

    assert exc_info.value.status_code == 403
    assert "Invalid request to AI model" in exc_info.value.detail
    assert "User does not have access to model" in exc_info.value.detail


def test_bedrock_404_error_should_raise_http_exception_404(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    error_response = {
        "Error": {
            "Code": "ResourceNotFoundException",
            "Message": "Model not found",
        },
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        side_effect=ClientError(error_response, "converse"),
    )

    with pytest.raises(HTTPException) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )

    assert exc_info.value.status_code == 404
    assert "Invalid request to AI model" in exc_info.value.detail
    assert "Model not found" in exc_info.value.detail


def test_bedrock_429_error_should_raise_http_exception_429(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    error_response = {
        "Error": {
            "Code": "ThrottlingException",
            "Message": "Rate exceeded",
        },
        "ResponseMetadata": {"HTTPStatusCode": 429},
    }
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        side_effect=ClientError(error_response, "converse"),
    )

    with pytest.raises(HTTPException) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )

    assert exc_info.value.status_code == 429
    assert "Invalid request to AI model" in exc_info.value.detail
    assert "Rate exceeded" in exc_info.value.detail


def test_bedrock_500_error_should_raise_http_exception_500(
    mocker: MockerFixture,
    bedrock_inference_service: service.BedrockInferenceService,
):
    error_response = {
        "Error": {
            "Code": "InternalServerException",
            "Message": "Internal service error",
        },
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        side_effect=ClientError(error_response, "converse"),
    )

    with pytest.raises(HTTPException) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="This is not a real prompt",
            messages=[
                {"role": "user", "content": [{"text": "What is the weather today?"}]}
            ],
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "AI model request failed"
