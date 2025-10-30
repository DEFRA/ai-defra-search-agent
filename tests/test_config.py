"""Test configuration and fixtures."""

import os
from pathlib import Path

import pytest


def load_env_file(env_file_path: Path) -> None:
    """Load environment variables from a file."""
    if not env_file_path.exists():
        return

    with env_file_path.open() as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                if key and value:
                    os.environ[key] = value


def pytest_configure(config):  # noqa: ARG001
    """Configure pytest - this runs before any tests are collected."""
    # Load test environment variables from .env.test file
    test_env_file = Path(__file__).parent.parent / ".env.test"
    load_env_file(test_env_file)


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment variables."""
    # Environment is already loaded in pytest_configure
    # This fixture just ensures cleanup
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
