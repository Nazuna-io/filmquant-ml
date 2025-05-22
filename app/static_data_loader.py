import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Get logger
logger = logging.getLogger("filmquant_ml.data_loader")

# Define the base path to the data directory relative to this file
# Assuming this file is in app/ and data is in ../data/
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# File paths for the sample data
GENRES_FILE = "sample_genres.json"
CAST_AND_CREW_FILE = "sample_cast_and_crew.json"
HISTORICAL_FILMS_FILE = "sample_historical_films.json"
STUDIOS_FILE = "sample_studios.json"

_genres_data = []
_cast_and_crew_data = []
_historical_films_data = []
_studios_data = []

def _safe_path_join(base_dir: str, *paths: str) -> str:
    """
    Safely join paths, preventing directory traversal attacks.
    
    Args:
        base_dir: Base directory that should contain the final path
        *paths: Path components to join
        
    Returns:
        Absolute path within the base directory
        
    Raises:
        ValueError: If the resulting path would be outside the base directory
    """
    # Resolve the base directory to an absolute path
    base_dir = os.path.abspath(base_dir)
    
    # Join the paths
    joined_path = os.path.abspath(os.path.join(base_dir, *paths))
    
    # Check if the joined path is within the base directory
    if not joined_path.startswith(base_dir):
        logger.error(f"Attempted directory traversal: {joined_path}")
        raise ValueError(f"Path would escape from base directory: {joined_path}")
    
    return joined_path

def _load_json_data(file_name: str) -> List[Dict[str, Any]]:
    """
    Helper function to load JSON data from a file safely.
    
    Args:
        file_name: Name of the file to load (not a path)
        
    Returns:
        List of data from the JSON file
    """
    # Create safe path
    try:
        file_path = _safe_path_join(DATA_DIR, file_name)
    except ValueError as e:
        logger.error(f"Security error: {str(e)}")
        return []
    
    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

def load_all_data():
    """Loads all reference data into memory."""
    global _genres_data, _cast_and_crew_data, _historical_films_data, _studios_data
    
    _genres_data = _load_json_data(GENRES_FILE)
    _cast_and_crew_data = _load_json_data(CAST_AND_CREW_FILE)
    _historical_films_data = _load_json_data(HISTORICAL_FILMS_FILE)
    _studios_data = _load_json_data(STUDIOS_FILE)
    
    logger.info(f"Loaded {len(_genres_data)} genres.")
    logger.info(f"Loaded {len(_cast_and_crew_data)} cast and crew members.")
    logger.info(f"Loaded {len(_historical_films_data)} historical films.")
    logger.info(f"Loaded {len(_studios_data)} studios.")

def get_genres():
    """Returns the loaded genres data."""
    return _genres_data

def get_cast_and_crew():
    """Returns the loaded cast and crew data."""
    return _cast_and_crew_data

def get_historical_films():
    """Returns the loaded historical films data."""
    return _historical_films_data

def get_studios():
    """Returns the loaded studios data."""
    return _studios_data

# Load data when the module is imported
load_all_data()

if __name__ == '__main__':
    # For testing purposes
    print("Static Data Loader Module Test")
    print("-----------------------------")
    print(f"Genres: {get_genres()[:2]}")
    print(f"Cast/Crew: {get_cast_and_crew()[:2]}")
    print(f"Historical Films: {get_historical_films()[:1]}")
    print(f"Studios: {get_studios()[:2]}")
