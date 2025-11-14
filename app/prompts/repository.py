import abc


class AbstractPromptRepository(abc.ABC):
    @abc.abstractmethod
    def get_prompt_by_name(self, name: str) -> str:
        pass
