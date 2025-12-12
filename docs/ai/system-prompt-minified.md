You must follow the Python AI coding rules in `docs/ai/ai-guidelines-python.md` whenever generating or modifying code in this repository.

Key constraints:
- All Python commands use `uv`; do not call python/pytest/ruff directly.
- Tests and linting must be run via `taskipy` tasks (e.g. `uv run task test`, `uv run task lint`).
- Testing uses pytest only: function-based tests, pytest fixtures (`mocker`, `tmp_path`, `monkeypatch`), no unittest.
- Style and formatting enforced by `ruff`; code must comply with the repo’s configuration.
- Comments must be minimal; prefer clear naming and small focused functions.
- Do not add dependencies or change public APIs unless explicitly instructed.

When modifying code:
- Follow existing patterns in the surrounding files.
- Add or update pytest tests for any changed behavior.
- Recommend appropriate `uv run task` commands.
