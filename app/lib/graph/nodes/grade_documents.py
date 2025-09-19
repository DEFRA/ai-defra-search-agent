from app.lib.graph.chains.retrieval_grader import retrieval_grader
from app.lib.graph.state import State


def grade_documents(state: State):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Filtered out irrelevent documents.
    """
    question = state["question"]
    context = state["context"]

    print(f"###### GRADING {len(context)} documents")

    filtered_documents = []

    for doc in context:
        score = retrieval_grader().invoke(
            {"document": doc.page_content, "question": question}
        )
        grade = score.binary_score
        print(f"Document: {doc.page_content} \nScore: {grade}")

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(doc)
        else:
            print("---GRADE: DOCUMENT IS IRRELEVANT---")
            continue

    return {"documents_for_context": filtered_documents}
