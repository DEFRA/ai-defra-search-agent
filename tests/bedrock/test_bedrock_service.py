import pytest

from app.bedrock import models, service

from tests.fixtures import bedrock as bedrock_fixtures


@pytest.fixture
def bedrock_inference_service():
    return service.BedrockInferenceService(
        api_client=bedrock_fixtures.StubBedrockClient(),
        runtime_client=bedrock_fixtures.StubBedrockRuntimeClient(),
    )


def test_no_guardrails_should_return_bedrock_response(
    bedrock_inference_service: service.BedrockInferenceService,
):
    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(id="geni-ai-3.5"),
        system_prompt="This is not a real prompt",
        messages=[{"role": "user", "content": "What is the weather today?"}],
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
        messages=[{"role": "user", "content": "What is the weather today?"}],
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
        messages=[{"role": "user", "content": "What is the weather today?"}],
    )

    assert response.model_id == "geni-ai-3.5"
    assert response.content == [{"text": "This is a stub response."}]


def test_invalid_guardrail_arn_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(ValueError) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5", guardrail_id="invalid-arn", guardrail_version=1
            ),
            system_prompt="This is not a real prompt",
            messages=[{"role": "user", "content": "What is the weather today?"}],
        )

    assert "Invalid guardrail ARN format" in str(exc_info.value)


def test_invalid_guardrail_version_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(ValueError) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5",
                guardrail_id="arn:aws:bedrock:us-west-2:123456789012:guardrail/8etdsfsdf3sd",
                guardrail_version=0,
            ),
            system_prompt="This is not a real prompt",
            messages=[{"role": "user", "content": "What is the weather today?"}],
        )

    assert "Guardrail version must be a positive integer" in str(exc_info.value)


def test_guardrail_id_with_no_version_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(Exception) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5",
                guardrail_id="arn:aws:bedrock:eu-central-1:123445511111:guardrail/8xqdsfsdf3gk",
                guardrail_version=None,
            ),
            system_prompt="This is not a real prompt",
            messages=[{"role": "user", "content": "What is the weather today?"}],
        )

    assert "The guardrail ID and version must be provided together" in str(
        exc_info.value
    )


def test_guardrail_version_with_no_id_should_raise_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(ValueError) as exc_info:
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(
                id="geni-ai-3.5", guardrail_id=None, guardrail_version=1
            ),
            system_prompt="This is not a real prompt",
            messages=[{"role": "user", "content": "What is the weather today?"}],
        )

    assert "The guardrail ID and version must be provided together" in str(
        exc_info.value
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
    with pytest.raises(ValueError) as exc_info:
        bedrock_inference_service.get_inference_profile_details(
            inference_profile_id="geni-ai-3.5"
        )

    assert "Invalid inference profile ID format" in str(exc_info.value)
