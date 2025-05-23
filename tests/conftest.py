# tests/conftest.py
import os
import sys

import pytest

# Add the project root to the Python path to allow imports from 'app'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.main import app as flask_app  # The Flask app instance after Gradio mounting
from app.static_data_loader import load_all_data


@pytest.fixture(scope="session")
def app():
    """Create and configure a new app instance for each test session."""
    # Ensure data is loaded for tests that might rely on it indirectly via app context
    load_all_data()
    flask_app.config.update(
        {
            "TESTING": True,
        }
    )
    yield flask_app


@pytest.fixture()
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands (if any)."""
    return app.test_cli_runner()
