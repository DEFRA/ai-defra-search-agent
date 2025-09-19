from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import config as settings
from app.lib.bedrock_client import chat_bedrock_client

GRADING_MODEL = settings.AWS_BEDROCK_MODEL_GRADING

llm = chat_bedrock_client(GRADING_MODEL)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an llm generation is grounded in /supported by a set of documents. \n
        Give a binary score of 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {context} \n\n LLM generation: {answer}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
