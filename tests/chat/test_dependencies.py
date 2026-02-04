import pytest
from pytest_mock import MockerFixture

from app.bedrock import service as bedrock_service
from app.chat import agent, dependencies, repository, service
from app.common import knowledge


def test_get_knowledge_retriever(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.knowledge.base_url = "http://knowledge-base.com"
    mock_config.knowledge.similarity_threshold = 0.5

    retriever = dependencies.get_knowledge_retriever(app_config=mock_config)

    assert isinstance(retriever, knowledge.KnowledgeRetriever)
    assert retriever.base_url == "http://knowledge-base.com"
    assert retriever.similarity_threshold == 0.5


def test_get_bedrock_runtime_client_no_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = False
    mock_config.sqs.region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_runtime_client(app_config=mock_config)

    mock_boto3.assert_called_with("bedrock-runtime", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_runtime_client_with_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.sqs.region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_runtime_client(app_config=mock_config)

    mock_boto3.assert_called_with(
        "bedrock-runtime",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",  # noqa: S106
        region_name="us-east-1",
    )
    assert client == mock_boto3.return_value


def test_get_bedrock_client_no_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = False
    mock_config.sqs.region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_client(app_config=mock_config)

    mock_boto3.assert_called_with("bedrock", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_client_with_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.sqs.region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_client(app_config=mock_config)

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
    mock_config = mocker.Mock()

    service_instance = dependencies.get_bedrock_inference_service(
        api_client=mock_client,
        runtime_client=mock_runtime_client,
        app_config=mock_config,
    )

    assert isinstance(service_instance, bedrock_service.BedrockInferenceService)
    assert service_instance.api_client == mock_client
    assert service_instance.runtime_client == mock_runtime_client
    assert service_instance.app_config == mock_config


def test_get_chat_agent(mocker: MockerFixture):
    mock_inference_service = mocker.Mock(spec=bedrock_service.BedrockInferenceService)
    mock_config = mocker.Mock()
    mock_prompt_repository = mocker.Mock()
    mock_prompt_repository.get_prompt_by_name.return_value = "Test system prompt"

    agent_instance = dependencies.get_chat_agent(
        mock_inference_service, mock_config, mock_prompt_repository
    )

    assert isinstance(agent_instance, agent.BedrockChatAgent)
    assert agent_instance.inference_service == mock_inference_service
    assert agent_instance.app_config == mock_config
    assert agent_instance.system_prompt == "Test system prompt"
    mock_prompt_repository.get_prompt_by_name.assert_called_once_with("system_prompt")


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


def test_get_sqs_client(mocker: MockerFixture):
    mock_sqs_client = mocker.patch("app.common.sqs.SQSClient")
    # call the factory to trigger construction but don't assign unused variable
    dependencies.get_sqs_client()

    mock_sqs_client.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_worker_services_monkeypatched_variation(mocker):
    """Alternate initialize_worker_services path with monkeypatched heavy deps."""
    # Patch mongo.get_mongo_client to return an object with __getitem__
    mocker.patch("app.chat.dependencies.mongo.get_mongo_client", autospec=True)
    from app.chat import dependencies as deps

    async def fake_get_mongo_client(_):
        class Dummy:
            def __getitem__(self, name):
                class DummyDB:
                    pass

                db = DummyDB()
                db.conversations = object()
                return db

        return Dummy()

    mocker.patch(
        "app.chat.dependencies.mongo.get_mongo_client",
        side_effect=fake_get_mongo_client,
    )
    mocker.patch("boto3.client")
    mocker.patch("app.prompts.repository.FileSystemPromptRepository")
    mocker.patch("app.chat.dependencies.get_chat_agent")
    mocker.patch("app.models.service.ConfigModelResolutionService")
    mocker.patch("app.common.sqs.SQSClient")

    # Provide a minimal config
    cfg = mocker.Mock()
    cfg.mongo.database = "db"
    cfg.bedrock.use_credentials = False
    cfg.sqs.region = "eu-1"
    cfg.knowledge.base_url = "http://k"
    cfg.knowledge.similarity_threshold = 0.5
    mocker.patch("app.chat.dependencies.config.get_config", return_value=cfg)

    chat_svc, conv_repo, sqs_client = await deps.initialize_worker_services()
    assert chat_svc is not None
    assert conv_repo is not None
    assert sqs_client is not None


@pytest.mark.asyncio
async def test_initialize_worker_services(mocker: MockerFixture):
    # Mock all the dependencies
    mock_get_config = mocker.patch("app.chat.dependencies.config.get_config")
    mock_get_mongo_client = mocker.patch("app.chat.dependencies.mongo.get_mongo_client")
    mocker.patch("app.chat.dependencies.get_knowledge_retriever")
    mocker.patch("boto3.client")  # Mock boto3.client to avoid AWS calls
    mocker.patch("app.prompts.repository.FileSystemPromptRepository")
    mocker.patch("app.chat.dependencies.get_chat_agent")
    mocker.patch("app.models.service.ConfigModelResolutionService")
    mock_sqs_client = mocker.patch("app.common.sqs.SQSClient")

    # Mock config object
    mock_config = mocker.Mock()
    mock_config.database.uri = "mongodb://localhost:27017"
    mock_config.database.name = "test_db"
    mock_config.mongo.database = "test_db"
    mock_config.sqs.region = "us-east-1"
    mock_config.bedrock.use_credentials = False
    mock_config.knowledge.base_url = "http://knowledge"
    mock_config.knowledge.similarity_threshold = 0.5
    mock_get_config.return_value = mock_config

    # Mock MongoDB client and database
    mock_client = mocker.MagicMock()
    mock_db = mocker.Mock()
    mock_client.__getitem__.return_value = mock_db
    mock_get_mongo_client.return_value = mock_client

    # Call the async function
    (
        chat_svc,
        conversation_repo,
        sqs_client,
    ) = await dependencies.initialize_worker_services()

    # Verify it returns the right types
    assert isinstance(chat_svc, service.ChatService)
    assert isinstance(conversation_repo, repository.MongoConversationRepository)

    # Verify key services were called
    mock_get_mongo_client.assert_called_once_with(mock_config)
    mock_sqs_client.assert_called_once()
