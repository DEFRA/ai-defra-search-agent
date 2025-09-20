from langchain_core.documents import Document


class State(dict):
    question: str
    context: dict[Document]
    documents_for_context: dict[Document]
    answer: str
    conversation_history: list | None = None
