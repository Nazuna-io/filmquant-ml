# app/data_ingestion/pipeline.py
"""
Base pipeline and data processing classes for the data ingestion pipeline.
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.utils.logging import get_logger

logger = get_logger("filmquant_ml.data_ingestion.pipeline")


class DataProcessor(ABC):
    """
    Abstract base class for data processors that can be chained in a pipeline.
    Each processor should implement the process method to transform the input data.
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"filmquant_ml.data_ingestion.{self.name}")

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data and return the transformed data.

        Args:
            data: Input DataFrame to process

        Returns:
            Transformed DataFrame
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make the processor callable for easy chaining.

        Args:
            data: Input DataFrame to process

        Returns:
            Transformed DataFrame
        """
        self.logger.info(f"Processing data with {self.name}")
        return self.process(data)


class DataIngestionPipeline:
    """
    Pipeline for data ingestion, preprocessing, and transformation.
    Chains multiple data processors together to process input data.
    """

    def __init__(
        self,
        processors: List[DataProcessor] = None,
        name: str = "DataIngestionPipeline",
    ):
        """
        Initialize a new data ingestion pipeline.

        Args:
            processors: List of data processors to apply in order
            name: Name of the pipeline for logging
        """
        self.processors = processors or []
        self.name = name
        self.logger = get_logger(f"filmquant_ml.data_ingestion.{self.name}")

    def add_processor(self, processor: DataProcessor) -> None:
        """
        Add a processor to the pipeline.

        Args:
            processor: Data processor to add
        """
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.name}")

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the pipeline on the input data.

        Args:
            data: Input DataFrame to process

        Returns:
            Processed DataFrame
        """
        self.logger.info(
            f"Starting pipeline run with {len(self.processors)} processors"
        )
        result = data.copy()

        for i, processor in enumerate(self.processors):
            self.logger.info(
                f"Running processor {i+1}/{len(self.processors)}: {processor.name}"
            )
            try:
                result = processor(result)
                self.logger.info(f"Processor {processor.name} completed successfully")
            except Exception as e:
                self.logger.error(
                    f"Error in processor {processor.name}: {str(e)}", exc_info=True
                )
                raise

        self.logger.info("Pipeline run completed successfully")
        return result

    def _ensure_valid_path(self, file_path: str) -> str:
        """
        Ensure a file path is valid and safe.

        Args:
            file_path: Path to validate

        Returns:
            Absolute path

        Raises:
            ValueError: If the path is invalid or unsafe
        """
        try:
            # Make path absolute
            abs_path = os.path.abspath(file_path)

            # Check if the parent directory exists
            parent_dir = os.path.dirname(abs_path)
            if not os.path.exists(parent_dir):
                self.logger.info(f"Creating directory: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)

            return abs_path
        except Exception as e:
            self.logger.error(f"Invalid file path: {file_path}, error: {str(e)}")
            raise ValueError(f"Invalid file path: {file_path}")

    def save_processed_data(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save the processed data to a CSV file.

        Args:
            data: Processed DataFrame to save
            output_path: Path to save the data to
        """
        # Ensure the path is valid and safe
        safe_path = self._ensure_valid_path(output_path)

        # Determine output format based on file extension
        _, ext = os.path.splitext(safe_path)
        ext = ext.lower()

        try:
            if ext == ".csv":
                self.logger.info(f"Saving processed data as CSV to {safe_path}")
                data.to_csv(safe_path, index=False)
            elif ext == ".json":
                self.logger.info(f"Saving processed data as JSON to {safe_path}")
                with open(safe_path, "w") as f:
                    json.dump(data.to_dict(orient="records"), f, indent=2)
            else:
                # Default to CSV if extension is not recognized
                self.logger.warning(
                    f"Unrecognized file extension: {ext}, defaulting to CSV"
                )
                if not safe_path.endswith(".csv"):
                    safe_path += ".csv"
                data.to_csv(safe_path, index=False)

            self.logger.info(f"Data saved successfully: {len(data)} rows")
        except Exception as e:
            self.logger.error(
                f"Error saving data to {safe_path}: {str(e)}", exc_info=True
            )
            raise
