from langchain_core.prompts import ChatPromptTemplate

from app.lib.aws_bedrock.bedrock_client import chat_bedrock
from app.lib.graph.state import State


def generate_final_answer(state: State) -> dict[str, any]:
    from app.prompts.loader import load_prompt

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("final_answer")),
            ("human", "{question}"),
        ]
    )

    messages = final_prompt.invoke(
        {"question": state["question"], "answer": state["answer"]}
    )

    response = chat_bedrock(messages, callback=None)

    print("###### GENERATED FINAL ANSWER")
    print(f"Original answer: {state['answer']}")
    print(f"Formatted answer: {response.content}")
    print("###### END GENERATED FINAL ANSWER")

    state["original_answer"] = state["answer"]
    state["answer"] = response.content

    return {"answer": response.content, "original_answer": state["answer"]}
