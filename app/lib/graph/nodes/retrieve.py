from app.lib.graph.state import State
from app.lib.store.vectorstore_client import VectorStoreClient


def retrieve(state: State) -> dict[str, any]:
    client = VectorStoreClient()
    retriever = client.as_retriever()
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}
