import abc
import pathlib

from langgraph import graph

from app.common import bedrock
from app.conversation_history import models as conv_models
from app.v2_chat import nodes, repository, state_models


class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(self, question: str, conversation: conv_models.ConversationHistory) -> state_models.ChatState:
        pass


class LangGraphChatAgent(AbstractChatAgent):
    def __init__(self):
        self._setup_graph()

    def _setup_graph(self):
        dependencies = nodes.NodeDependencies(
            inference_service=bedrock.BedrockInferenceService(),
            prompt_repository=repository.FileSystemPromptRepository(prompt_directory=f"{pathlib.Path(__file__).parent}/prompts")
        )

        node_container = nodes.GraphNodes(dependencies)

        workflow = graph.StateGraph(
            state_models.ChatState,
            input_schema=state_models.InputState,
            output_schema=state_models.OutputState
        )

        workflow.add_node(nodes.GraphNodes.RETRIEVE, node_container.retrieve)
        workflow.add_node(nodes.GraphNodes.GRADE_DOCUMENTS, node_container.grade_documents)
        workflow.add_node(nodes.GraphNodes.GENERATE, node_container.generate)
        workflow.add_node(nodes.GraphNodes.FINAL_ANSWER, node_container.format_final_answer)

        workflow.add_edge(nodes.GraphNodes.RETRIEVE, nodes.GraphNodes.GRADE_DOCUMENTS)
        workflow.add_edge(nodes.GraphNodes.GRADE_DOCUMENTS, nodes.GraphNodes.GENERATE)
        workflow.add_edge(nodes.GraphNodes.GENERATE, nodes.GraphNodes.FINAL_ANSWER)
        workflow.add_edge(nodes.GraphNodes.FINAL_ANSWER, graph.END)

        workflow.set_entry_point(nodes.GraphNodes.RETRIEVE)

        self._app = workflow.compile()

    async def execute_flow(self, question: str, conversation: conv_models.ConversationHistory) -> state_models.OutputState:
        state = state_models.ChatState(
            question=question,
            conversation_history=conversation.messages
        )

        return await self._app.ainvoke(state)
