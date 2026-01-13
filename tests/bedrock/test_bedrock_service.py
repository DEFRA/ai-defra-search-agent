from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from app.bedrock import models, service


@pytest.fixture
def bedrock_inference_service(
    mocker: MockerFixture, bedrock_client, bedrock_runtime_v2_client
):
    mock_config = mocker.Mock()
    mock_config.bedrock.max_response_tokens = 100
    mock_config.bedrock.default_model_temprature = 0.5
    mock_knowledge_retriever = mocker.Mock()
    mock_knowledge_retriever._filter_relevant_docs = lambda x: x

    return service.BedrockInferenceService(
        api_client=bedrock_client,
        runtime_client=bedrock_runtime_v2_client,
        app_config=mock_config,
        knowledge_retriever=mock_knowledge_retriever,
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
        model_config=models.ModelConfig(id="geni-ai-3.5"),
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
            guardrail_version="1",
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
                id="geni-ai-3.5", guardrail_id=None, guardrail_version="1"
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


def test_invoke_anthropic_with_empty_messages_raises_error(
    bedrock_inference_service: service.BedrockInferenceService,
):
    with pytest.raises(
        ValueError, match="Cannot invoke Anthropic model with no messages"
    ):
        bedrock_inference_service.invoke_anthropic(
            model_config=models.ModelConfig(id="geni-ai-3.5"),
            system_prompt="prompt",
            messages=[],
        )


def test_invoke_with_rag_should_augment_prompt_and_return_sources(
    bedrock_inference_service: service.BedrockInferenceService,
    mocker: MockerFixture,
):
    mock_retriever = cast(Any, bedrock_inference_service.knowledge_retriever)
    mock_retriever.search.return_value = [
        {
            "name": "doc1",
            "location": "http://doc1",
            "content": "This is content of doc1",
            "similarity_score": 0.9,
        }
    ]

    # Mock runtime client converse to check prompt
    mock_converse = mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        },
    )
    # Ensure _get_backing_model returns something valid so we don't fail there
    mocker.patch.object(
        bedrock_inference_service, "_get_backing_model", return_value="geni-ai-3.5"
    )

    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(id="geni-ai-3.5"),
        system_prompt="System prompt.",
        messages=[{"role": "user", "content": [{"text": "Query"}]}],
        knowledge_group_id="group1",
    )

    # Check that search was called
    mock_retriever.search.assert_called_with(group_id="group1", query="Query")

    # Check that system prompt in converse call contains the doc content
    _, kwargs = mock_converse.call_args
    system_prompts = kwargs["system"]
    assert len(system_prompts) == 1
    full_prompt = system_prompts[0]["text"]
    assert "System prompt." in full_prompt
    assert "<context>" in full_prompt
    assert "This is content of doc1" in full_prompt

    # Check sources in response
    assert len(response.sources) == 1
    assert response.sources[0].name == "doc1"
    assert response.sources[0].location == "http://doc1"


def test_invoke_with_rag_but_no_docs_found(
    bedrock_inference_service: service.BedrockInferenceService,
    mocker: MockerFixture,
):
    # Mock retrieval to return empty list
    mock_retriever = cast(Any, bedrock_inference_service.knowledge_retriever)
    mock_retriever.search.return_value = []

    mock_converse = mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        },
    )

    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(id="geni-ai-3.5"),
        system_prompt="System prompt.",
        messages=[{"role": "user", "content": [{"text": "Query"}]}],
        knowledge_group_id="group1",
    )

    # Check that system prompt is NOT modified
    _, kwargs = mock_converse.call_args
    system_prompts = kwargs["system"]
    assert len(system_prompts) == 1
    full_prompt = system_prompts[0]["text"]
    assert full_prompt == "System prompt."  # Unchanged
    assert response.sources == []


def test_retrieve_knowledge_returns_empty_when_no_retriever(
    mocker: MockerFixture,
    bedrock_client,
    bedrock_runtime_v2_client,
):
    # Create service without knowledge_retriever
    mock_config = mocker.Mock()
    svc = service.BedrockInferenceService(
        api_client=bedrock_client,
        runtime_client=bedrock_runtime_v2_client,
        app_config=mock_config,
        knowledge_retriever=None,
    )

    result = svc._retrieve_knowledge(
        messages=[{"role": "user", "content": [{"text": "Query"}]}],
        knowledge_group_id="group1",
    )

    assert result == []


def test_invoke_with_inference_profile_arn_should_resolve_backing_model(
    bedrock_inference_service: service.BedrockInferenceService,
    mocker: MockerFixture,
):
    profile_arn = "arn:aws:bedrock:us-west-2:123456789012:inference-profile/my-profile"
    backing_model_arn = "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"

    # Mock get_inference_profile_details to return a profile with a model
    mocker.patch.object(
        bedrock_inference_service,
        "get_inference_profile_details",
        return_value=models.InferenceProfile(
            id=profile_arn,
            name="My Profile",
            models=[{"modelArn": backing_model_arn}],
        ),
    )

    # Mock runtime client
    mocker.patch.object(
        bedrock_inference_service.runtime_client,
        "converse",
        return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        },
    )

    response = bedrock_inference_service.invoke_anthropic(
        model_config=models.ModelConfig(id=profile_arn),
        system_prompt="System prompt.",
        messages=[{"role": "user", "content": [{"text": "Query"}]}],
    )

    assert response.model_id == profile_arn
    assert response.content == [{"text": "Response"}]
    assert response.usage == {"input_tokens": 10, "output_tokens": 20}
    assert response.sources == []
