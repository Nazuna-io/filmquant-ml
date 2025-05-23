"""
Configuration validation for FilmQuant ML.

This module provides functions to validate configuration settings using Dynaconf validators.
"""

from dynaconf import Validator

from app.config.settings import settings

# List of validators for configuration settings
validators = [
    # App section
    Validator("app.name", must_exist=True, is_type_of=str),
    Validator("app.version", must_exist=True, is_type_of=str),
    Validator("app.debug", must_exist=True, is_type_of=bool),
    Validator("app.host", must_exist=True, is_type_of=str),
    Validator("app.port", must_exist=True, is_type_of=int),
    # Logging section
    Validator("logging.level", must_exist=True, is_type_of=str),
    Validator("logging.format", must_exist=True, is_type_of=str),
    Validator("logging.directory", must_exist=True, is_type_of=str),
    Validator("logging.console", must_exist=True, is_type_of=bool),
    Validator("logging.file", must_exist=True, is_type_of=bool),
    Validator("logging.max_bytes", must_exist=True, is_type_of=int),
    Validator("logging.backup_count", must_exist=True, is_type_of=int),
    # API section
    Validator("api.prefix", must_exist=True, is_type_of=str),
    Validator("api.cors_origins", must_exist=True, is_type_of=list),
    Validator("api.rate_limit", must_exist=True, is_type_of=int),
    # Data section
    Validator("data.reference_data_dir", must_exist=True, is_type_of=str),
    Validator("data.model_dir", must_exist=True, is_type_of=str),
    # Data ingestion section
    Validator("data_ingestion.default_config_path", must_exist=True, is_type_of=str),
    # Evaluation section
    Validator("evaluation.metrics_dir", must_exist=True, is_type_of=str),
    Validator("evaluation.plots_dir", must_exist=True, is_type_of=str),
    Validator("evaluation.model_tracking_dir", must_exist=True, is_type_of=str),
]


def validate_settings():
    """
    Validate all settings against defined validators.

    Raises:
        dynaconf.validator.ValidationError: If validation fails
    """
    settings.validators.register(*validators)
    settings.validators.validate()

    return True
