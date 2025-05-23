#!/usr/bin/env python3
"""Test script to verify that our configuration module is working correctly."""
from app.config import get, settings


def main():
    """Print out configuration settings."""
    print("APP SETTINGS:")
    print(f"  Name: {settings.app.name}")
    print(f"  Version: {settings.app.version}")
    print(f"  Debug: {settings.app.debug}")
    print(f"  Host: {settings.app.host}")
    print(f"  Port: {settings.app.port}")

    print("\nLOGGING SETTINGS:")
    print(f"  Level: {settings.logging.level}")
    print(f"  Format: {settings.logging.format}")
    print(f"  Directory: {settings.logging.directory}")
    print(f"  Console: {settings.logging.console}")
    print(f"  File: {settings.logging.file}")

    print("\nAPI SETTINGS:")
    print(f"  Prefix: {settings.api.prefix}")
    print(f"  CORS Origins: {settings.api.cors_origins}")
    print(f"  Rate Limit: {settings.api.rate_limit}")

    print("\nEVALUATION SETTINGS:")
    print(f"  Metrics Dir: {settings.evaluation.metrics_dir}")
    print(f"  Plots Dir: {settings.evaluation.plots_dir}")
    print(f"  Model Tracking Dir: {settings.evaluation.model_tracking_dir}")

    print("\nUsing get() function:")
    print(f"  App Name: {get('app.name')}")
    print(f"  Debug Mode: {get('app.debug')}")
    print(
        f"  Non-existent Key (with default): {get('non_existent.key', 'default_value')}"
    )

    print("\nLoaded Configuration Files:")
    for file_path in settings.get_loaded_files():
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
