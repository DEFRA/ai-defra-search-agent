from langchain_core.prompts import ChatPromptTemplate

from app.lib.aws_bedrock.bedrock_client import chat_bedrock
from app.lib.graph.state import State


def generate_final_answer(state: State) -> dict[str, any]:
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are given a text that contains a correct and factual answer. Do not change the meaning or add new information. Reformat the answer so that it renders correctly in a .njk template using UK Government

                    Specifically:

                    1. Present the main points as <ul class="govuk-list govuk-list--bullet"> bullet points.
                    2. Add a short summary in a <p class="govuk-body"> paragraph at the top.
                    3. Ensure all text uses GOV.UK Frontend classes (e.g. govuk-body, govuk-list, etc.).
                    4. Do not include extra boilerplate — only the formatted content block.
                    5. Ensure the content is clear, concise, and professional, suitable for a government audience.
                    Here is the text to reformat:
                    {answer}

            """,
            ),
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
