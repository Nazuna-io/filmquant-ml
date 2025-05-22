# app/data_ingestion/ingest.py
"""
Data ingestion script for FilmQuant ML.
"""
import os
import argparse
import logging
import pandas as pd
import json
from typing import List, Dict, Any, Optional

from app.utils.logging import configure_logging, get_logger
from app.data_ingestion.pipeline import DataIngestionPipeline
from app.data_ingestion.processors import (
    CleaningProcessor,
    DateFeatureProcessor,
    CategoricalEncodingProcessor,
    NumericalFeatureProcessor,
    DataSplitProcessor,
    FeatureSelectionProcessor
)

# Configure logging
logger = configure_logging(
    app_name="filmquant_ml.data_ingestion",
    log_level=logging.INFO
)

def create_pipeline(config: Dict[str, Any]) -> DataIngestionPipeline:
    """
    Create a data ingestion pipeline based on the provided configuration.
    
    Args:
        config: Configuration dictionary with pipeline settings
        
    Returns:
        Configured DataIngestionPipeline instance
    """
    pipeline = DataIngestionPipeline(name="FilmQuantMLDataIngestionPipeline")
    
    # Add processors based on configuration
    if config.get('cleaning', {}).get('enabled', True):
        cleaning_config = config.get('cleaning', {})
        pipeline.add_processor(
            CleaningProcessor(
                drop_na_columns=cleaning_config.get('drop_na_columns', ['budget_usd', 'box_office_revenue']),
                fill_na_values=cleaning_config.get('fill_na_values', {})
            )
        )
    
    if config.get('date_features', {}).get('enabled', True):
        date_config = config.get('date_features', {})
        pipeline.add_processor(
            DateFeatureProcessor(
                date_columns=date_config.get('date_columns', ['release_date']),
                output_format=date_config.get('output_format', 'onehot')
            )
        )
    
    if config.get('categorical_encoding', {}).get('enabled', True):
        cat_config = config.get('categorical_encoding', {})
        pipeline.add_processor(
            CategoricalEncodingProcessor(
                categorical_columns=cat_config.get('categorical_columns', ['genre_ids', 'studio_id']),
                method=cat_config.get('method', 'onehot'),
                target_column=cat_config.get('target_column', 'box_office_revenue')
            )
        )
    
    if config.get('numerical_features', {}).get('enabled', True):
        num_config = config.get('numerical_features', {})
        pipeline.add_processor(
            NumericalFeatureProcessor(
                numerical_columns=num_config.get('numerical_columns', ['budget_usd', 'runtime_minutes']),
                transformations=num_config.get('transformations', {'log': ['budget_usd', 'box_office_revenue']})
            )
        )
    
    if config.get('data_split', {}).get('enabled', True):
        split_config = config.get('data_split', {})
        pipeline.add_processor(
            DataSplitProcessor(
                target_column=split_config.get('target_column', 'box_office_revenue'),
                test_size=split_config.get('test_size', 0.2),
                random_state=split_config.get('random_state', 42),
                split_method=split_config.get('split_method', 'random'),
                split_column=split_config.get('split_column', None)
            )
        )
    
    if config.get('feature_selection', {}).get('enabled', True):
        feat_config = config.get('feature_selection', {})
        pipeline.add_processor(
            FeatureSelectionProcessor(
                target_column=feat_config.get('target_column', 'box_office_revenue'),
                method=feat_config.get('method', 'correlation'),
                threshold=feat_config.get('threshold', 0.05),
                max_features=feat_config.get('max_features', None),
                include_columns=feat_config.get('include_columns', [])
            )
        )
    
    return pipeline

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV or JSON file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    # Validate and make path absolute
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            logger.info(f"Loading CSV data from: {file_path}")
            df = pd.read_csv(file_path)
        elif file_ext == '.json':
            logger.info(f"Loading JSON data from: {file_path}")
            # Assume JSON is an array of objects, not a single object
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}, expected .csv or .json")
        
        logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {str(e)}")
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path}: {str(e)}")
        raise ValueError(f"CSV parsing error in {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise

def main(input_path: str, output_path: str, config_path: str = None):
    """
    Main function for the data ingestion script.
    
    Args:
        input_path: Path to the input data file
        output_path: Path to save the processed data
        config_path: Path to the pipeline configuration file (optional)
    """
    logger.info(f"Starting data ingestion process")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Config: {config_path if config_path else 'Using default configuration'}")
    
    # Load configuration if provided, otherwise use defaults
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            config = {}
    else:
        config = {}
    
    try:
        # Load data
        data = load_data(input_path)
        
        # Create and run pipeline
        pipeline = create_pipeline(config)
        processed_data = pipeline.run(data)
        
        # Save processed data
        pipeline.save_processed_data(processed_data, output_path)
        
        logger.info(f"Data ingestion completed successfully")
        logger.info(f"Processed data saved to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilmQuant ML Data Ingestion Pipeline")
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the input data file (CSV or JSON)"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="Path to save the processed data (CSV)"
    )
    parser.add_argument(
        "--config", "-c", 
        default=None, 
        help="Path to the pipeline configuration file (JSON)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    success = main(args.input, args.output, args.config)
    exit(0 if success else 1)
