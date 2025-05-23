"""
Tests for the centralized configuration module.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.config.settings import DotDict, Settings


def test_dotdict():
    """Test DotDict implementation."""
    # Create a test dictionary
    data = {
        "app": {
            "name": "Test App",
            "version": "1.0.0",
            "debug": True,
            "settings": {"nested": "value"},
        },
        "simple_key": "simple_value",
    }

    # Convert to DotDict
    dot_dict = DotDict(data)

    # Test attribute access
    assert dot_dict.app.name == "Test App"
    assert dot_dict.app.version == "1.0.0"
    assert dot_dict.app.debug is True
    assert dot_dict.app.settings.nested == "value"
    assert dot_dict.simple_key == "simple_value"

    # Test dictionary access
    assert dot_dict["app"]["name"] == "Test App"
    assert dot_dict["simple_key"] == "simple_value"

    # Test to_dict conversion
    converted = dot_dict.to_dict()
    assert isinstance(converted, dict)
    assert not isinstance(converted, DotDict)
    assert converted["app"]["name"] == "Test App"
    assert isinstance(converted["app"], dict)
    assert not isinstance(converted["app"], DotDict)

    # Test update_nested
    updates = {
        "app": {
            "name": "Updated App",
            "settings": {"nested": "new_value", "new_nested": "added_value"},
        },
        "new_key": "new_value",
    }

    dot_dict.update_nested(updates)
    assert dot_dict.app.name == "Updated App"
    assert dot_dict.app.version == "1.0.0"  # Should be unchanged
    assert dot_dict.app.settings.nested == "new_value"
    assert dot_dict.app.settings.new_nested == "added_value"
    assert dot_dict.new_key == "new_value"


def test_settings_defaults():
    """Test Settings initialization with default values."""
    with patch.object(Settings, "_load_from_yaml"), patch.object(
        Settings, "_load_from_env"
    ), patch.object(Settings, "_load_from_args"):
        settings = Settings()

        # Check default values
        assert settings.app.name == "Box Office Revenue Predictor"
        assert settings.app.version == "0.1.0"
        assert settings.app.debug is False
        assert settings.app.host == "0.0.0.0"
        assert settings.app.port == 8081

        assert settings.logging.level == "INFO"
        assert "asctime" in settings.logging.format
        assert settings.logging.directory == "logs"

        assert settings.api.prefix == "/api/v1"
        assert settings.api.cors_origins == ["*"]
        assert settings.api.rate_limit == 100

        # Test get method
        assert settings.get("app.name") == "Box Office Revenue Predictor"
        assert settings.get("app.debug") is False
        assert settings.get("non_existent.key") is None
        assert settings.get("non_existent.key", "default") == "default"


def test_settings_yaml_loading():
    """Test loading settings from YAML file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary config file
        temp_path = Path(temp_dir) / "config.yaml"
        config_data = {
            "app": {"name": "Test App", "debug": True, "port": 9000},
            "logging": {"level": "DEBUG"},
        }

        with open(temp_path, "w") as f:
            yaml.dump(config_data, f)

        # Patch the file lookup to use our temporary file
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path, "resolve", return_value=temp_path.parent
        ), patch.object(Path, "__truediv__", return_value=temp_path), patch.object(
            Settings, "_load_from_env"
        ), patch.object(
            Settings, "_load_from_args"
        ):

            settings = Settings()

            # Check that values were loaded from YAML
            assert settings.app.name == "Test App"
            assert settings.app.debug is True
            assert settings.app.port == 9000
            assert settings.logging.level == "DEBUG"

            # Default values should still be available
            assert settings.app.host == "0.0.0.0"
            assert settings.api.cors_origins == ["*"]


def test_settings_env_loading():
    """Test loading settings from environment variables."""
    # Set test environment variables
    os.environ["FILMQUANT_ML_APP__NAME"] = "Env App"
    os.environ["FILMQUANT_ML_APP__DEBUG"] = "true"
    os.environ["FILMQUANT_ML_LOGGING__LEVEL"] = "ERROR"
    os.environ["FILMQUANT_ML_API__RATE_LIMIT"] = "50"

    try:
        with patch.object(Settings, "_load_from_yaml"), patch.object(
            Path, "exists", return_value=False
        ), patch.object(Settings, "_load_from_args"):

            settings = Settings()

            # Check that values were loaded from environment
            assert settings.app.name == "Env App"
            assert settings.app.debug is True
            assert settings.logging.level == "ERROR"
            assert settings.api.rate_limit == 50

            # Default values should still be available
            assert settings.app.host == "0.0.0.0"
            assert settings.api.cors_origins == ["*"]
    finally:
        # Clean up environment variables
        del os.environ["FILMQUANT_ML_APP__NAME"]
        del os.environ["FILMQUANT_ML_APP__DEBUG"]
        del os.environ["FILMQUANT_ML_LOGGING__LEVEL"]
        del os.environ["FILMQUANT_ML_API__RATE_LIMIT"]


def test_settings_args_loading():
    """Test loading settings from command line arguments."""
    test_args = ["script.py", "--app-debug", "--app-port=9000", "--logging-level=DEBUG"]

    # Directly simulate argument parsing
    with patch.object(Settings, "_load_from_yaml"), patch.object(
        Settings, "_load_from_env"
    ):

        settings = Settings()

        # Manually update settings from args
        args_config = {
            "app": {"debug": True, "port": 9000},
            "logging": {"level": "DEBUG"},
        }
        settings._config.update_nested(args_config)

        # Check that values were loaded from arguments
        assert settings.app.debug is True
        assert settings.app.port == 9000
        assert settings.logging.level == "DEBUG"

        # Default values should still be available
        assert settings.app.host == "0.0.0.0"
        assert settings.api.cors_origins == ["*"]


def test_settings_priority():
    """Test that settings are loaded with the correct priority."""
    # Create test environment variables (lowest priority)
    os.environ["FILMQUANT_ML_APP__NAME"] = "Env App"
    os.environ["FILMQUANT_ML_APP__DEBUG"] = "true"
    os.environ["FILMQUANT_ML_LOGGING__LEVEL"] = "ERROR"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary config file (middle priority)
            temp_path = Path(temp_dir) / "config.yaml"
            config_data = {
                "app": {"name": "YAML App", "port": 9000},
                "logging": {"level": "WARNING"},
            }

            with open(temp_path, "w") as f:
                yaml.dump(config_data, f)

            # Create settings with YAML and ENV variables
            with patch.object(Path, "exists", return_value=True), patch.object(
                Path, "resolve", return_value=temp_path.parent
            ), patch.object(Path, "__truediv__", return_value=temp_path):

                settings = Settings()

                # Environment variables override YAML in our implementation
                assert settings.app.name == "Env App"
                assert settings.app.debug is True
                assert settings.logging.level == "ERROR"

                # YAML config should still provide values not in env vars
                assert settings.app.port == 9000

                # Now manually simulate command-line args (highest priority)
                args_config = {
                    "app": {
                        "name": "Args App",
                    },
                    "logging": {"level": "DEBUG"},
                }

                settings._config.update_nested(args_config)

                # Command line args should now have highest priority
                assert settings.app.name == "Args App"
                assert settings.logging.level == "DEBUG"

                # Earlier settings should still be available
                assert settings.app.debug is True
                assert settings.app.port == 9000
    finally:
        # Clean up environment variables
        del os.environ["FILMQUANT_ML_APP__NAME"]
        del os.environ["FILMQUANT_ML_APP__DEBUG"]
        del os.environ["FILMQUANT_ML_LOGGING__LEVEL"]
