from abc import ABC, abstractmethod
from pathlib import Path

from langgraph.graph import END, StateGraph

from app.common.bedrock import BedrockInferenceService
from app.conversation_history.models import ConversationHistory
from app.v2_chat.nodes import GraphNodes, NodeDependencies
from app.v2_chat.repository import FileSystemPromptRepository
from app.v2_chat.state_models import ChatState, InputState, OutputState


class AbstractChatAgent(ABC):
    @abstractmethod
    async def execute_flow(self, question: str, conversation: ConversationHistory) -> ChatState:
        pass


class LangGraphChatAgent(AbstractChatAgent):
    def __init__(self):
        self._setup_graph()

    def _setup_graph(self):
        dependencies = NodeDependencies(
            inference_service=BedrockInferenceService(),
            prompt_repository=FileSystemPromptRepository(prompt_directory=f"{Path(__file__).parent}/prompts")
        )

        node_container = GraphNodes(dependencies)

        workflow = StateGraph(
            ChatState,
            input_schema=InputState,
            output_schema=OutputState
        )

        workflow.add_node(GraphNodes.RETRIEVE, node_container.retrieve)
        workflow.add_node(GraphNodes.GRADE_DOCUMENTS, node_container.grade_documents)
        workflow.add_node(GraphNodes.GENERATE, node_container.generate)

        workflow.add_edge(GraphNodes.RETRIEVE, GraphNodes.GRADE_DOCUMENTS)
        workflow.add_edge(GraphNodes.GRADE_DOCUMENTS, GraphNodes.GENERATE)
        workflow.add_edge(GraphNodes.GENERATE, END)

        workflow.set_entry_point(GraphNodes.RETRIEVE)

        self._app = workflow.compile()

    async def execute_flow(self, question: str, conversation: ConversationHistory) -> OutputState:
        state = ChatState(
            question=question,
            conversation_history=conversation.messages
        )

        return await self._app.ainvoke(state)
