from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import config as settings
from app.lib.aws_bedrock.bedrock_client import chat_bedrock_client

GRADING_MODEL = settings.AWS_BEDROCK_MODEL_GRADING


class GradeDocument(BaseModel):
    """Binary score for relevance check on the retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevent to the question. 'yes or 'no'."
    )


system = """You are a grader assessing the relevence of a retrieved document to a question. \n
If the document contains keyword(s) or sematic meaning related to the question, grade it as relevant. \n
Give a binary score of 'yes' or 'no' to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved document: \n\n {document} \n\n User question: {question}",
        ),
    ]
)


def retrieval_grader():
    """Create and return the retrieval grader chain."""
    llm = chat_bedrock_client(GRADING_MODEL)
    structured_llm_grader = llm.with_structured_output(GradeDocument)
    return grade_prompt | structured_llm_grader
