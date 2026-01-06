import pytest

from app.prompts.repository import FileSystemPromptRepository


class TestFileSystemPromptRepository:
    def test_init_creates_repository_with_valid_directory(self, tmp_path):
        """Test that repository initializes with a valid directory"""
        repo = FileSystemPromptRepository(prompts_dir=tmp_path)
        assert repo.prompts_dir == tmp_path
        assert repo._cache == {}

    def test_init_raises_error_for_nonexistent_directory(self, tmp_path):
        """Test that repository raises error when directory doesn't exist"""
        nonexistent_path = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="Prompts directory does not exist"):
            FileSystemPromptRepository(prompts_dir=nonexistent_path)

    def test_init_raises_error_for_non_directory(self, tmp_path):
        """Test that repository raises error when path is not a directory"""
        file_path = tmp_path / "not_a_directory.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="Prompts path is not a directory"):
            FileSystemPromptRepository(prompts_dir=file_path)

    def test_get_prompt_by_name_loads_from_file(self, tmp_path):
        """Test that get_prompt_by_name loads content from file"""
        # Create a test prompt file
        prompt_content = "This is a test system prompt."
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        # Initialize repository and load prompt
        repo = FileSystemPromptRepository(prompts_dir=tmp_path)
        result = repo.get_prompt_by_name("test_prompt")

        assert result == prompt_content

    def test_get_prompt_by_name_caches_content(self, tmp_path):
        """Test that prompt content is cached after first load"""
        prompt_content = "This is a test system prompt."
        prompt_file = tmp_path / "cached_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        repo = FileSystemPromptRepository(prompts_dir=tmp_path)

        # First call - loads from file
        result1 = repo.get_prompt_by_name("cached_prompt")
        assert result1 == prompt_content
        assert "cached_prompt" in repo._cache

        # Modify the file
        prompt_file.write_text("Modified content", encoding="utf-8")

        # Second call - should return cached content, not modified content
        result2 = repo.get_prompt_by_name("cached_prompt")
        assert result2 == prompt_content  # Still returns original content

    def test_get_prompt_by_name_raises_error_for_missing_file(self, tmp_path):
        """Test that get_prompt_by_name raises FileNotFoundError for missing file"""
        repo = FileSystemPromptRepository(prompts_dir=tmp_path)

        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            repo.get_prompt_by_name("nonexistent_prompt")

    def test_get_prompt_by_name_strips_whitespace(self, tmp_path):
        """Test that get_prompt_by_name strips leading/trailing whitespace"""
        prompt_content = "  \n  Test prompt with whitespace  \n  "
        prompt_file = tmp_path / "whitespace_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        repo = FileSystemPromptRepository(prompts_dir=tmp_path)
        result = repo.get_prompt_by_name("whitespace_prompt")

        assert result == "Test prompt with whitespace"

    def test_clear_cache_removes_all_cached_prompts(self, tmp_path):
        """Test that clear_cache removes all cached prompts"""
        # Create test files
        (tmp_path / "prompt1.txt").write_text("Prompt 1", encoding="utf-8")
        (tmp_path / "prompt2.txt").write_text("Prompt 2", encoding="utf-8")

        repo = FileSystemPromptRepository(prompts_dir=tmp_path)

        # Load prompts to cache them
        repo.get_prompt_by_name("prompt1")
        repo.get_prompt_by_name("prompt2")
        assert len(repo._cache) == 2

        # Clear cache
        repo.clear_cache()
        assert len(repo._cache) == 0

    def test_get_cached_prompt_names_returns_cached_names(self, tmp_path):
        """Test that get_cached_prompt_names returns list of cached prompt names"""
        # Create test files
        (tmp_path / "prompt1.txt").write_text("Prompt 1", encoding="utf-8")
        (tmp_path / "prompt2.txt").write_text("Prompt 2", encoding="utf-8")

        repo = FileSystemPromptRepository(prompts_dir=tmp_path)

        # Initially, no cached prompts
        assert repo.get_cached_prompt_names() == []

        # Load prompts
        repo.get_prompt_by_name("prompt1")
        repo.get_prompt_by_name("prompt2")

        # Check cached names
        cached_names = repo.get_cached_prompt_names()
        assert len(cached_names) == 2
        assert "prompt1" in cached_names
        assert "prompt2" in cached_names

    def test_multiple_prompts_independent_cache(self, tmp_path):
        """Test that multiple different prompts are cached independently"""
        (tmp_path / "prompt_a.txt").write_text("Content A", encoding="utf-8")
        (tmp_path / "prompt_b.txt").write_text("Content B", encoding="utf-8")

        repo = FileSystemPromptRepository(prompts_dir=tmp_path)

        result_a = repo.get_prompt_by_name("prompt_a")
        result_b = repo.get_prompt_by_name("prompt_b")

        assert result_a == "Content A"
        assert result_b == "Content B"
        assert len(repo._cache) == 2
