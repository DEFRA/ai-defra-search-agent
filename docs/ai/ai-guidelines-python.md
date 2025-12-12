# AI Coding Guidelines (Python)

These rules apply to all AI-assisted Python development in this repository.

The AI assistant MUST follow these guidelines when generating or modifying Python code, tests, or tooling.

---

## 1. Packaging, Tasks & Dependency Management

- The Python package manager is **`uv`**.
- All Python commands MUST be executed via `uv`.
- Operational commands (tests, lint, formatting, etc.) MUST be run via **`taskipy`** tasks configured in `pyproject.toml`.

### 1.1 Command patterns

The assistant MUST:

- Use `uv run task <name>` for project tasks such as testing and linting.
- Prefer existing tasks over invoking tools directly.

Examples:

- Run tests:
  - `uv run task test`
- Run lint + format:
  - `uv run task lint`

Only when there is **no appropriate task** defined in `pyproject.toml` is it acceptable to suggest a direct tool invocation like:

- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format .`

The assistant MUST:

- Avoid suggesting `pip`, `virtualenv`, `conda`, `poetry`, or `pipenv` unless explicitly requested.
- Avoid suggesting raw `python` / `pytest` / `ruff` commands without `uv run`.

When proposing **new recurring workflows** (e.g. a type-checking step), the assistant SHOULD:

- Suggest adding a new `taskipy` task in `pyproject.toml`, e.g.:

  ```toml
  [tool.taskipy.tasks]
  typecheck = "mypy src"
  ```

- Then run it via `uv run task typecheck`.

---

## 2. Testing

### 2.1 General

- The organisation requires **> 90% unit test coverage**.
- Every non-trivial change SHOULD include or update tests.
- New modules and public functions SHOULD have corresponding tests.

The assistant SHOULD:

- Propose tests along with implementation.
- Update existing tests when behavior changes.
- Highlight visible coverage gaps when appropriate.

### 2.2 Test Framework

- **Testing framework:** `pytest`.
- Do **NOT** use the `unittest` framework or subclass `unittest.TestCase`.

The assistant MUST:

- Write pytest-style tests:
  - Test files named like `test_*.py` or `*_test.py`.
  - Tests as **functions**, not classes.
- Prefer **function-based tests** over class-based tests.

### 2.3 Pytest Idioms & Fixtures

The assistant SHOULD use standard pytest features and idioms:

- Built-in fixtures:
  - `tmp_path` for temporary filesystem interactions.
  - `monkeypatch` for modifying environment/attributes.
- The `mocker` fixture (if `pytest-mock` is available in this project) for mocking.

The assistant MUST:

- Prefer pytest fixtures and idioms over custom temp-dir or mocking code.
- Prefer `mocker.patch` over `unittest.mock.patch`.

Tests SHOULD be runnable via:

- `uv run task test`

---

## 3. Style, Linting & Comments

### 3.1 Style & Linting

- Follow **PEP 8** and idiomatic Python style.
- Use **`ruff`** as the single source of truth for linting and formatting rules.

The assistant MUST:

- Generate code expected to pass `ruff`.
- Prefer clarity over cleverness.

Linting/formatting SHOULD be run via:

- `uv run task lint`

Only if no lint task exists, use:

- `uv run ruff check .`
- `uv run ruff format .`

### 3.2 Imports & Structure

- Use absolute imports unless the repo clearly prefers relative ones.
- Import order:
  1. Standard library
  2. Third-party
  3. Internal modules

### 3.3 Commenting Guidelines

Comments MUST be minimal.

The assistant SHOULD:

- Avoid adding comments that merely restate what the code already expresses.
- Prefer **clear naming**, **small focused functions**, and **readable logic** instead of verbose explanations.
- Add comments **only when absolutely necessary** to aid understanding:
  - non-obvious decisions  
  - subtle edge cases  
  - justification for unusual patterns  

Comments SHOULD explain **why**, not **what**.

---

## 4. Behaviour When Modifying Code

When modifying or adding code, the assistant SHOULD:

1. **Match existing patterns** in nearby modules.
2. Keep functions small, explicit, and readable.
3. Preserve public APIs unless instructed otherwise.
4. Add or update tests in the appropriate test module.
5. Recommend running:
   - `uv run task test`
   - `uv run task lint`

---

## 5. Forbidden / Strongly Discouraged Practices

The assistant MUST NOT:

- Introduce alternate package managers or dependency systems.
- Use `unittest` or `unittest.TestCase`.
- Suggest disabling `ruff` or adding broad `# noqa` comments.
- Introduce new third-party dependencies unless explicitly instructed.
- Suggest running raw `pytest` or `ruff` without using `uv` and `taskipy` where applicable.

The assistant SHOULD avoid:

- Large refactors without tests.
- Over-engineering or unnecessary abstractions.

---

## 6. Self-checklist for AI-generated Python Changes

Before finalising, the assistant SHOULD verify:

- [ ] All suggested commands use `uv` and `taskipy` when applicable.
- [ ] Tests use pytest, function-based, no unittest.
- [ ] Pytest fixtures used appropriately (`mocker`, `tmp_path`, `monkeypatch`, etc.).
- [ ] Code is expected to pass `ruff` checks.
- [ ] No new dependencies added without instruction.
- [ ] Public APIs preserved unless explicitly changed.
- [ ] Tests added or updated for new or changed behaviour.

