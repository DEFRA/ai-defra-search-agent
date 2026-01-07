import abc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "resources" / "prompts"


class AbstractPromptRepository(abc.ABC):
    @abc.abstractmethod
    def get_prompt_by_name(self, name: str) -> str:
        pass


class FileSystemPromptRepository(AbstractPromptRepository):
    def __init__(self):
        self.prompts_dir = PROMPTS_DIR
        self._cache: dict[str, str] = {}

        if not self.prompts_dir.exists():
            msg = f"Prompts directory does not exist: {self.prompts_dir}"
            raise FileNotFoundError(msg)

        if not self.prompts_dir.is_dir():
            msg = f"Prompts path is not a directory: {self.prompts_dir}"
            raise ValueError(msg)

    def get_prompt_by_name(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]

        prompt_file = self.prompts_dir / f"{name}.txt"

        if not prompt_file.exists():
            msg = f"Prompt file not found: {prompt_file}"
            raise FileNotFoundError(msg)

        logger.info("Loading prompt '%s' from file: %s", name, prompt_file)

        try:
            with open(prompt_file, encoding="utf-8") as f:
                prompt_content = f.read().strip()

            self._cache[name] = prompt_content

            return prompt_content
        except Exception as e:
            msg = f"Error reading prompt file {prompt_file}: {e}"
            raise RuntimeError(msg) from e

    def clear_cache(self) -> None:
        self._cache.clear()
