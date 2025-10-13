from dataclasses import dataclass

from app.common.bedrock import BedrockInferenceService
from app.common.http_client import create_async_client
from app.config import config
from app.v2_chat.repository import AbstractPromptRepository
from app.v2_chat.state_models import ChatState, KnowledgeDocument


@dataclass
class NodeDependencies:
    inference_service: BedrockInferenceService
    prompt_repository: AbstractPromptRepository


class GraphNodes:
    RETRIEVE = "retrieve"
    GRADE_DOCUMENTS = "grade_documents"
    GENERATE = "generate"
    FINAL_ANSWER = "final_answer"

    def __init__(self, dependencies: NodeDependencies):
        self.dependencies = dependencies

    async def retrieve(self, state: ChatState) -> ChatState:
        request = {
            "groupId": config.workflow.default_knowledge_group_id,
            "query": state.question,
            "maxResults": 5
        }

        async with create_async_client() as client:
            response = await client.post(f"{config.workflow.data_service_url}snapshots/query", json=request)
            response.raise_for_status()

            documents = [
                KnowledgeDocument(
                    content=doc.get("content", ""),
                    snapshot_id=doc.get("snapshotId", ""),
                    source_id=doc.get("sourceId", ""),
                    metadata=doc.get("metadata", {})
                )
                for doc in response.json()
            ]

            return { "candidate_documents": documents }

    def grade_documents(self, state: ChatState) -> ChatState:
        prompt = self.dependencies.prompt_repository.get_prompt_by_name("retrieval_grader.txt")

        context = []

        for doc in state.candidate_documents:
            grade_result = self.dependencies.inference_service.invoke_anthropic(
                config.bedrock.grading_model,
                system_prompt=prompt,
                messages=[{"role": "user", "content": f"Retrieved document: {doc.content}"}]
            )

            if grade_result.content[-1]["text"].strip().lower() == "yes":
                context.append(doc)

        return { "context": context }

    def generate(self, state: ChatState) -> ChatState:
        prompt = self.dependencies.prompt_repository.get_prompt_by_name("qa_system.txt")

        if len(state.context) == 0:
            return { "answer": "I'm sorry, I couldn't find any relevant information to answer your question." }

        joined_context = "\n\n".join([doc.content for doc in state.context])

        full_prompt = prompt.format(context=joined_context)

        # Include the user's question as a message
        messages = [{"role": "user", "content": state.question}]

        generation = self.dependencies.inference_service.invoke_anthropic(
            config.bedrock.generation_model,
            system_prompt=full_prompt,
            messages=messages
        )

        answer = generation.content[0]["text"]

        return { "answer": answer }
