"""Simple test script for the configuration module."""
import os
import sys
import tempfile
from pathlib import Path
import yaml

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config.settings import Settings, DotDict

def test_dotdict():
    """Test DotDict implementation."""
    # Create a test dictionary
    data = {
        "app": {
            "name": "Test App",
            "version": "1.0.0",
            "debug": True,
            "settings": {
                "nested": "value"
            }
        },
        "simple_key": "simple_value"
    }
    
    # Convert to DotDict
    dot_dict = DotDict(data)
    
    # Test attribute access
    assert dot_dict.app.name == "Test App"
    assert dot_dict.app.version == "1.0.0"
    assert dot_dict.app.debug is True
    assert dot_dict.app.settings.nested == "value"
    assert dot_dict.simple_key == "simple_value"
    
    print("DotDict test passed!")

def test_settings_load():
    """Test settings basic loading."""
    settings = Settings()
    
    print("\nSettings loaded:")
    print(f"App Name: {settings.app.name}")
    print(f"App Debug: {settings.app.debug}")
    print(f"Logging Level: {settings.logging.level}")
    
    print("\nLoaded files:")
    for file_path in settings.get_loaded_files():
        print(f"- {file_path}")
    
    print("\nSettings load test passed!")

def test_env_variables():
    """Test environment variable loading."""
    # Set test environment variables
    os.environ["FILMQUANT_ML_APP__NAME"] = "Env Test App"
    os.environ["FILMQUANT_ML_APP__DEBUG"] = "true"
    os.environ["FILMQUANT_ML_LOGGING__LEVEL"] = "DEBUG"
    
    try:
        # Reload settings
        settings = Settings()
        
        # Check values
        assert settings.app.name == "Env Test App"
        assert settings.app.debug is True
        assert settings.logging.level == "DEBUG"
        
        print("\nEnvironment variables test passed!")
    finally:
        # Clean up
        del os.environ["FILMQUANT_ML_APP__NAME"] 
        del os.environ["FILMQUANT_ML_APP__DEBUG"]
        del os.environ["FILMQUANT_ML_LOGGING__LEVEL"]

def main():
    """Run all tests."""
    print("Running configuration module tests...")
    
    test_dotdict()
    test_settings_load()
    test_env_variables()
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
