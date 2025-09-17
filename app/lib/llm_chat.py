from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.lib.bedrock_client import chat_bedrock_client
from app.lib.vectorstore_client import VectorStoreClient


def run_llm(query: str, chat_history: list[tuple[dict, any]], chat_client=None):
    chat = chat_client or chat_bedrock_client()
    retriever = VectorStoreClient().as_retriever()

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=retriever, prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    callback = UsageMetadataCallbackHandler()

    result = qa.invoke(
        {"input": query, "chat_history": chat_history}, config={"callbacks": [callback]}
    )

    return {
        "question": result["input"],
        "answer": result["answer"],
        "source_documents": result["context"],
        "chat_history": chat_history,
        "usage": callback.usage_metadata,
    }
