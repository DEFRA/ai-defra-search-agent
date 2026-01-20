from pytest_mock import MockerFixture
from motor.motor_asyncio import AsyncIOMotorClient

from app.bedrock import service as bedrock_service
from app.chat import agent, dependencies, job_repository, repository, service
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
    mock_config.aws_region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_runtime_client(app_config=mock_config)

    mock_boto3.assert_called_with("bedrock-runtime", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_runtime_client_with_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.aws_region = "us-east-1"

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
    mock_config.aws_region = "us-east-1"

    mock_boto3 = mocker.patch("boto3.client")

    client = dependencies.get_bedrock_client(app_config=mock_config)

    mock_boto3.assert_called_with("bedrock", region_name="us-east-1")
    assert client == mock_boto3.return_value


def test_get_bedrock_client_with_credentials(mocker: MockerFixture):
    mock_config = mocker.Mock()
    mock_config.bedrock.use_credentials = True
    mock_config.bedrock.access_key_id = "test-key"
    mock_config.bedrock.secret_access_key = "test-secret"  # noqa: S105
    mock_config.aws_region = "us-east-1"

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

    client = dependencies.get_sqs_client()

    mock_sqs_client.assert_called_once()
    assert client == mock_sqs_client.return_value





def test_get_job_repository(mocker: MockerFixture):
    mock_db = mocker.Mock()
    mock_client = mocker.MagicMock()  # Use MagicMock to support __getitem__
    mock_database = mocker.Mock()
    
    # Configure subscripting
    mock_client.__getitem__.return_value = mock_database
    
    mock_db.client = mock_client
    mock_db.name = "test_db"

    repo = dependencies.get_job_repository(mock_db)

    assert isinstance(repo, job_repository.MongoJobRepository)
    mock_client.__getitem__.assert_called_once_with("test_db")


import pytest

@pytest.mark.asyncio
async def test_initialize_worker_services(mocker: MockerFixture):
    # Mock all the dependencies
    mock_get_config = mocker.patch("app.chat.dependencies.config.get_config")
    mock_get_mongo_client = mocker.patch("app.chat.dependencies.mongo.get_mongo_client")
    mock_get_knowledge_retriever = mocker.patch("app.chat.dependencies.get_knowledge_retriever")
    mock_boto3 = mocker.patch("boto3.client")  # Mock boto3.client to avoid AWS calls
    mock_prompt_repository = mocker.patch("app.prompts.repository.FileSystemPromptRepository")
    mock_get_chat_agent = mocker.patch("app.chat.dependencies.get_chat_agent")
    mock_model_service = mocker.patch("app.models.service.ConfigModelResolutionService")
    mock_sqs_client = mocker.patch("app.common.sqs.SQSClient")

    # Mock config object
    mock_config = mocker.Mock()
    mock_config.database.uri = "mongodb://localhost:27017"
    mock_config.database.name = "test_db"
    mock_config.mongo.database = "test_db"
    mock_config.aws_region = "us-east-1"
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
    chat_svc, job_repo, sqs_cli = await dependencies.initialize_worker_services()

    # Verify it returns the right types
    assert isinstance(chat_svc, service.ChatService)
    assert isinstance(job_repo, job_repository.MongoJobRepository)
    
    # Verify key services were called
    mock_get_mongo_client.assert_called_once_with(mock_config)
    mock_sqs_client.assert_called_once()
