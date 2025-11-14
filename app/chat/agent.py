import abc

from app.bedrock import service
from app.chat import models


class AbstractChatAgent(abc.ABC):
    @abc.abstractmethod
    async def execute_flow(
        self, question: str
    ) -> list[models.Message]:
        pass


class BedrockChatAgent(AbstractChatAgent):
    def __init__(self, inference_service: service.BedrockInferenceService):
        self.inference_service = inference_service

    async def execute_flow(
        self, conversation: models.Conversation
    ) -> list[models.Message]:
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in conversation.messages
        ]
        
        response = self.inference_service.invoke_anthropic(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_prompt="You are a helpful assistant.",
            messages=messages
        )
        
        # Extract text content from the response
        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].get("text", "")
        
        return [
            models.Message(
                role="assistant",
                content=content,
                model=response.model,
            )
        ]
