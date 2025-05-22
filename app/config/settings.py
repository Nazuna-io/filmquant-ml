"""
Centralized configuration module for FilmQuant ML.

This module provides a central configuration system for the application.
"""
import os
import yaml
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
from dotenv import load_dotenv

# Default configuration dictionary
DEFAULT_CONFIG = {
    "app": {
        "name": "Box Office Revenue Predictor",
        "version": "0.1.0",
        "debug": False,
        "host": "0.0.0.0",
        "port": 8081,
    },
    "logging": {
        "level": "INFO",
        "format": "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s",
        "directory": "logs",
        "console": True,
        "file": True,
        "max_bytes": 10485760,  # 10MB
        "backup_count": 10,
    },
    "api": {
        "prefix": "/api/v1",
        "cors_origins": ["*"],
        "rate_limit": 100,  # requests per minute
    },
    "data": {
        "reference_data_dir": "data",
        "model_dir": "data/models",
    },
    "data_ingestion": {
        "default_config_path": "config/data_ingestion_default.json",
    },
    "evaluation": {
        "metrics_dir": "evaluation_results/metrics",
        "plots_dir": "evaluation_results/plots",
        "model_tracking_dir": "evaluation_results/model_tracking",
    },
}

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

class DotDict(dict):
    """Dictionary that allows attribute-style access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dictionaries to DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def to_dict(self) -> dict:
        """Convert DotDict back to a regular dictionary."""
        result = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result
    
    def update_nested(self, updates: dict) -> None:
        """Update dictionary, allowing for nested updates."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in self and isinstance(self[key], DotDict):
                self[key].update_nested(value)
            else:
                self[key] = DotDict(value) if isinstance(value, dict) else value

class Settings:
    """
    Central configuration management class for FilmQuant ML.
    
    Loads configuration from:
    1. Default values
    2. YAML configuration file
    3. Environment variables
    4. Command-line arguments
    """
    
    def __init__(self):
        """Initialize settings with default values."""
        self._config = DotDict(DEFAULT_CONFIG)
        self._loaded_files = []
        
        # Load configuration in a specific order
        self._load_from_yaml()
        self._load_from_env()
        self._load_from_args()
        
        # Validate required settings in production mode
        self.validate_production_settings()
    
    def _load_from_yaml(self):
        """Load configuration from YAML file."""
        # Look for config.yaml in the project root
        project_root = Path(__file__).parents[2].resolve()
        config_path = project_root / "config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        self._config.update_nested(yaml_config)
                        self._loaded_files.append(str(config_path))
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {str(e)}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Load .env file if it exists
        env_file = Path(__file__).parents[2].resolve() / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            self._loaded_files.append(str(env_file))
        
        # Process environment variables with FILMQUANT_ML_ prefix
        prefix = "FILMQUANT_ML_"
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split by double underscore for nesting
                key_parts = key[len(prefix):].lower().split('__')
                
                # Convert value to appropriate type
                if value.lower() in ['true', 'yes', '1']:
                    value = True
                elif value.lower() in ['false', 'no', '0']:
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
                
                # Build nested dict structure
                current = env_config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the final value
                current[key_parts[-1]] = value
        
        # Update configuration with environment variables
        if env_config:
            self._config.update_nested(env_config)
    
    def _load_from_args(self):
        """Load configuration from command-line arguments."""
        # If not running from a script with arguments, skip
        if len(sys.argv) <= 1:
            return
            
        parser = argparse.ArgumentParser(description="FilmQuant ML Configuration")
        
        # Common configuration options
        parser.add_argument("--config", help="Path to configuration file")
        parser.add_argument("--app-debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--app-port", type=int, help="Port to listen on")
        parser.add_argument("--app-host", help="Host to bind to")
        parser.add_argument("--app-name", help="Application name")
        parser.add_argument("--logging-level", 
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          help="Logging level")
        
        # Parse known args to allow for other command line arguments
        args, _ = parser.parse_known_args()
        args_dict = vars(args)
        
        # Remove None values
        args_dict = {k: v for k, v in args_dict.items() if v is not None}
        
        # Handle custom config file
        if "config" in args_dict and args_dict["config"]:
            config_path = args_dict["config"]
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        custom_config = yaml.safe_load(f)
                        if custom_config:
                            self._config.update_nested(custom_config)
                            self._loaded_files.append(config_path)
                except Exception as e:
                    logging.warning(f"Failed to load config from {config_path}: {str(e)}")
        
        # Process other command line arguments
        args_config = {}
        for key, value in args_dict.items():
            if key == "config":
                continue
                
            # Split key by dash for nesting
            key_parts = key.split("-")
            
            # Build nested dict structure
            current = args_config
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value
            current[key_parts[-1]] = value
        
        # Update configuration with command line arguments
        if args_config:
            self._config.update_nested(args_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key in dot notation (e.g., "app.debug")
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        current = self._config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        
        return current
    
    def __getattr__(self, key: str) -> Any:
        """Allow direct attribute access for top-level sections."""
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"'Settings' object has no attribute '{key}'")
    
    # Environment variables with defaults for development, but required in production
    _REQUIRED_IN_PRODUCTION = [
        'APP_SECRET_KEY',
    ]
    
    def validate_production_settings(self):
        """
        Validate that all required settings are present in production mode.
        
        Raises:
            ConfigurationError: If a required setting is missing in production
        """
        # Check if we're in production mode
        if not self._config.get('app', {}).get('debug', False):
            missing_vars = []
            for var in self._REQUIRED_IN_PRODUCTION:
                env_var = f"FILMQUANT_ML_{var}"
                if env_var not in os.environ:
                    missing_vars.append(var)
            
            if missing_vars:
                raise ConfigurationError(
                    f"Missing required environment variables in production mode: {', '.join(missing_vars)}"
                )
            
        return True
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary."""
        return self._config.to_dict()
    
    def get_loaded_files(self) -> List[str]:
        """Get list of loaded configuration files."""
        return self._loaded_files

# Create and export a singleton settings instance
settings = Settings()

# Export common functions
get = settings.get
