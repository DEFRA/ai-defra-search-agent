import abc

class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(self, question: str, conversation: conv_models.ConversationHistory) -> state_models.ChatState:
        pass
