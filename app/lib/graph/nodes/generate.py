from langchain_core.prompts import ChatPromptTemplate

from app.lib.aws_bedrock.bedrock_client import chat_bedrock
from app.lib.graph.state import State


def format_conversation_history(history: list[dict]) -> str:
    """
    Format conversation history as a readable transcript for the LLM prompt.
    Shows last 10 turns (user/assistant pairs) for brevity.
    """
    if not history:
        return ""
    lines = []
    for msg in history[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        answer = msg.get("answer")
        if role == "user":
            lines.append(f"User: {content}")
            if answer:
                lines.append(f"Assistant: {answer}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def generate(state: State) -> dict[str, any]:
    from app.prompts.loader import load_prompt

    hardened_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("qa_system")),
            ("human", "{question}"),
        ]
    )

    docs_content = "\n\n".join(
        doc.page_content for doc in state["documents_for_context"]
    )

    if not docs_content.strip():
        return {
            "answer": "I don't have any relevant documents to answer your question. Please try rephrasing your query."
        }

    formatted_history = format_conversation_history(state["conversation_history"])

    messages = hardened_prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content,
            "conversation_history": formatted_history,
        }
    )

    response = chat_bedrock(messages, callback=None)
    return {"answer": response.content}
