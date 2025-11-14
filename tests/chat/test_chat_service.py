import pytest

from app.chat import models, repository, service
from tests.fixtures import agent as agent_fixtures

async def test_execute_chat_new_conversation(mongo):
    chat_service = service.ChatService(
        chat_agent=agent_fixtures.StubChatAgent(),
        conversation_repository=repository.MongoConversationRepository(mongo.db),
    )

    conversation = await chat_service.execute_chat("Hello")

    assert conversation is not None
    assert conversation.id is not None
    assert len(conversation.messages) == 2
    assert conversation.messages[0].content == "Hello"
    assert conversation.messages[1].content == "This is a stub response."


async def test_execute_chat_existing_conversation(mongo):
    chat_service = service.ChatService(
        chat_agent=agent_fixtures.StubChatAgent(),
        conversation_repository=repository.MongoConversationRepository(mongo.db),
    )

    conversation = await chat_service.execute_chat("Hello")
    conversation_id = conversation.id

    updated_conversation = await chat_service.execute_chat(
        "How are you?", conversation_id=conversation_id
    )

    assert updated_conversation is not None
    assert updated_conversation.id == conversation_id
    assert len(updated_conversation.messages) == 4
    assert updated_conversation.messages[2].content == "How are you?"
    assert updated_conversation.messages[3].content == "This is a stub response."


async def test_nonexistent_conversation_raises_error(mongo):
    chat_service = service.ChatService(
        chat_agent=agent_fixtures.StubChatAgent(),
        conversation_repository=repository.MongoConversationRepository(mongo.db),
    )


    mongo.db.conversations.delete_many({})

    with pytest.raises(models.ConversationNotFoundError):
        await chat_service.execute_chat("Hello", conversation_id="nonexistent")
