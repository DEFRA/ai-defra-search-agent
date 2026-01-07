import pytest

from app.prompts import repository
from app.prompts.repository import FileSystemPromptRepository


class TestFileSystemPromptRepository:
    def test_init_creates_repository_with_valid_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()
        assert repo.prompts_dir == tmp_path
        assert repo._cache == {}

    def test_init_raises_error_for_nonexistent_directory(self, tmp_path, monkeypatch):
        nonexistent_path = tmp_path / "does_not_exist"
        monkeypatch.setattr(repository, "PROMPTS_DIR", nonexistent_path)

        with pytest.raises(FileNotFoundError, match="Prompts directory does not exist"):
            FileSystemPromptRepository()

    def test_init_raises_error_for_non_directory(self, tmp_path, monkeypatch):
        file_path = tmp_path / "not_a_directory.txt"
        file_path.touch()
        monkeypatch.setattr(repository, "PROMPTS_DIR", file_path)

        with pytest.raises(ValueError, match="Prompts path is not a directory"):
            FileSystemPromptRepository()

    def test_get_prompt_by_name_loads_from_file(self, tmp_path, monkeypatch):
        prompt_content = "This is a test system prompt."
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()
        result = repo.get_prompt_by_name("test_prompt")

        assert result == prompt_content

    def test_get_prompt_by_name_caches_content(self, tmp_path, monkeypatch):
        prompt_content = "This is a test system prompt."
        prompt_file = tmp_path / "cached_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()

        result1 = repo.get_prompt_by_name("cached_prompt")
        assert result1 == prompt_content
        assert "cached_prompt" in repo._cache

        prompt_file.write_text("Modified content", encoding="utf-8")

        result2 = repo.get_prompt_by_name("cached_prompt")
        assert result2 == prompt_content

    def test_get_prompt_by_name_raises_error_for_missing_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()

        with pytest.raises(
            RuntimeError, match="Prompt 'nonexistent_prompt' not found in directory"
        ):
            repo.get_prompt_by_name("nonexistent_prompt")

    def test_get_prompt_by_name_strips_whitespace(self, tmp_path, monkeypatch):
        prompt_content = "  \n  Test prompt with whitespace  \n  "
        prompt_file = tmp_path / "whitespace_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()
        result = repo.get_prompt_by_name("whitespace_prompt")

        assert result == "Test prompt with whitespace"

    def test_clear_cache_removes_all_cached_prompts(self, tmp_path, monkeypatch):
        (tmp_path / "prompt1.txt").write_text("Prompt 1", encoding="utf-8")
        (tmp_path / "prompt2.txt").write_text("Prompt 2", encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()

        repo.get_prompt_by_name("prompt1")
        repo.get_prompt_by_name("prompt2")
        assert len(repo._cache) == 2

        repo.clear_cache()
        assert len(repo._cache) == 0

    def test_get_cached_prompt_names_returns_cached_names(self, tmp_path, monkeypatch):
        (tmp_path / "prompt1.txt").write_text("Prompt 1", encoding="utf-8")
        (tmp_path / "prompt2.txt").write_text("Prompt 2", encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()

        assert list(repo._cache.keys()) == []

        repo.get_prompt_by_name("prompt1")
        repo.get_prompt_by_name("prompt2")

        cached_names = list(repo._cache.keys())
        assert len(cached_names) == 2
        assert "prompt1" in cached_names
        assert "prompt2" in cached_names

    def test_multiple_prompts_independent_cache(self, tmp_path, monkeypatch):
        (tmp_path / "prompt_a.txt").write_text("Content A", encoding="utf-8")
        (tmp_path / "prompt_b.txt").write_text("Content B", encoding="utf-8")

        monkeypatch.setattr(repository, "PROMPTS_DIR", tmp_path)
        repo = FileSystemPromptRepository()

        result_a = repo.get_prompt_by_name("prompt_a")
        result_b = repo.get_prompt_by_name("prompt_b")

        assert result_a == "Content A"
        assert result_b == "Content B"
        assert len(repo._cache) == 2
