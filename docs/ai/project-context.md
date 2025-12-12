# Project Context for AI Agents

## Purpose of the System

The `ai-defra-search-agent` is a Python-based backend service that acts as the AI Assistant for the AI DEFRA Search application. It handles chat interactions and knowledge retrieval by leveraging AWS Bedrock for generative AI capabilities. The service is built using the FastAPI framework and utilizes LangChain and LangGraph to orchestrate AI agent workflows, manage conversation state, and integrate with underlying data sources (MongoDB).

## Architecture Overview

- `app/...` – Main application logic (FastAPI, LangChain/LangGraph agents).
  - `app/entrypoints/...` – Application entry points (e.g., `fastapi.py`).
  - `app/chat/...` – Core chat functionality, agent logic, and routing.
  - `app/bedrock/...` – AWS Bedrock integration and model handling.
  - `app/common/...` – Shared utilities (logging, metrics, HTTP client).
- `tests/...` – Comprehensive pytest suite (unit and integration tests).
- `scripts/...` – Command-line utilities and helper scripts (e.g., security checks).
- `docs/...` – Documentation including AI guidelines and system prompts.

## Key Constraints

- **Dependency Management**: All execution uses `uv` for dependency management and virtual environment handling.
- **Task Management**: All common development tasks (linting, testing) use `taskipy` as defined in `pyproject.toml`.
- **Code Quality**: All code must pass `ruff` for linting and formatting.
- **Testing**: All tests must use `pytest` function-style patterns.
- **Python Version**: The project requires Python >= 3.12.

## Example Files to Reference

- `app/entrypoints/fastapi.py` – Main application entry point.
- `app/chat/service.py` – Example of service layer logic.
- `app/chat/agent.py` – Example of LangGraph agent definition.
- `tests/chat/test_service.py` – Example of a pytest test file.
- `pyproject.toml` – Project configuration and dependencies.

