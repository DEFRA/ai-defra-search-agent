from logging import getLogger

from langgraph.graph import END, StateGraph

from app.lib.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE
from app.lib.graph.chains.hallucination_grader import hallucination_grader
from app.lib.graph.nodes import generate, grade_documents, retrieve
from app.lib.graph.state import State

logger = getLogger(__name__)


def grade_generation_grounded_in_documents(state: State) -> str:
    answer = state["answer"]
    context = state["documents_for_context"]

    score = hallucination_grader().invoke({"context": context, "answer": answer})

    if score.binary_score:
        print("---GRADE: ANSWER IS GROUNDED IN THE FACTS---")
        return "useful"

    return "not_useful"


workflow = StateGraph(State)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(GRADE_DOCUMENTS, GENERATE)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents,
    {
        "not useful": RETRIEVE,
        "useful": END,
        "not supported": RETRIEVE,
    },
)

workflow.add_edge(GENERATE, END)

app = workflow.compile()
