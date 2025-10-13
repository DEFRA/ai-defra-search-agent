from abc import ABC, abstractmethod


class AbstractPromptRepository(ABC):
    @abstractmethod
    def get_prompt_by_name(self, name: str) -> str:
        pass


class FileSystemPromptRepository(AbstractPromptRepository):
    def __init__(self, prompt_directory: str):
        self.prompt_directory = prompt_directory

    def get_prompt_by_name(self, name: str) -> str:
        try:
            with open(f"{self.prompt_directory}/{name}") as file:
                return file.read()
        except FileNotFoundError as err:
            msg = f"Prompt '{name}' not found in directory '{self.prompt_directory}'"
            raise RuntimeError(msg) from err
