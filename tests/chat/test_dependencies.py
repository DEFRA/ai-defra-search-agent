from pytest_mock import MockerFixture

from app.bedrock import service as bedrock_service
from app.chat import agent, dependencies, repository, service


def test_get_bedrock_runtime_client_no_credentials(mocker: MockerFixture):
    dependencies.get_bedrock_runtime_client.cache_clear()

    # Mock config
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = False
    mock_config.aws_region = "us-east-1"

    mocker.patch("app.chat.dependencies.app_config", mock_config)
    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_runtime_client()

    mock_boto3.assert_called_with("bedrock-runtime", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_runtime_client_with_credentials(mocker: MockerFixture):
    dependencies.get_bedrock_runtime_client.cache_clear()

    # Mock config
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.aws_region = "us-east-1"

    mocker.patch("app.chat.dependencies.app_config", mock_config)
    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_runtime_client()

    mock_boto3.assert_called_with(
        "bedrock-runtime",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",  # noqa: S106
        region_name="us-east-1",
    )
    assert client == mock_boto3.return_value


def test_get_bedrock_client_no_credentials(mocker: MockerFixture):
    dependencies.get_bedrock_client.cache_clear()

    # Mock config
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = False
    mock_config.aws_region = "us-east-1"

    mocker.patch("app.chat.dependencies.app_config", mock_config)
    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_client()

    mock_boto3.assert_called_with("bedrock", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_client_with_credentials(mocker: MockerFixture):
    dependencies.get_bedrock_client.cache_clear()

    # Mock config
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.aws_region = "us-east-1"

    mocker.patch("app.chat.dependencies.app_config", mock_config)
    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_client()

    mock_boto3.assert_called_with(
        "bedrock",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",  # noqa: S106
        region_name="us-east-1",
    )
    assert client == mock_boto3.return_value


def test_get_bedrock_inference_service(mocker: MockerFixture):
    mock_client = mocker.Mock()
    mock_runtime_client = mocker.Mock()

    mocker.patch("app.chat.dependencies.get_bedrock_client", return_value=mock_client)
    mocker.patch(
        "app.chat.dependencies.get_bedrock_runtime_client",
        return_value=mock_runtime_client,
    )

    service_instance = dependencies.get_bedrock_inference_service()

    assert isinstance(service_instance, bedrock_service.BedrockInferenceService)
    assert service_instance.api_client == mock_client
    assert service_instance.runtime_client == mock_runtime_client


def test_get_chat_agent(mocker: MockerFixture):
    mock_inference_service = mocker.Mock(spec=bedrock_service.BedrockInferenceService)

    agent_instance = dependencies.get_chat_agent(mock_inference_service)

    assert isinstance(agent_instance, agent.BedrockChatAgent)
    assert agent_instance.inference_service == mock_inference_service


def test_get_conversation_repository(mocker: MockerFixture):
    mock_db = mocker.Mock()

    repo = dependencies.get_conversation_repository(mock_db)

    assert isinstance(repo, repository.MongoConversationRepository)


def test_get_chat_service(mocker: MockerFixture):
    mock_agent = mocker.Mock()
    mock_repo = mocker.Mock()

    chat_service = dependencies.get_chat_service(mock_agent, mock_repo)

    assert isinstance(chat_service, service.ChatService)
    assert chat_service.chat_agent == mock_agent
    assert chat_service.conversation_repository == mock_repo
