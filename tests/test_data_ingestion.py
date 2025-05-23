# tests/test_data_ingestion.py
"""
Tests for the data ingestion pipeline.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.data_ingestion.pipeline import DataIngestionPipeline, DataProcessor
from app.data_ingestion.processors import (
    CategoricalEncodingProcessor,
    CleaningProcessor,
    DateFeatureProcessor,
    NumericalFeatureProcessor,
)


class MockProcessor(DataProcessor):
    """Mock processor for testing the pipeline."""

    def __init__(self, name="MockProcessor", raise_error=False):
        super().__init__(name=name)
        self.processed = False
        self.raise_error = raise_error

    def process(self, data):
        """Process the data."""
        if self.raise_error:
            raise ValueError("Mock processing error")

        self.processed = True
        return data


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Film A", "Film B", "Film C", "Film D", "Film E"],
            "budget_usd": [100000, 200000, 300000, 400000, None],
            "box_office_revenue": [500000, 600000, None, 800000, 900000],
            "release_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
                "2020-05-01",
            ],
            "genre": ["Action", "Comedy", "Drama", "Sci-Fi", "Comedy"],
            "runtime_minutes": [90, 120, 150, 180, 120],
        }
    )


def test_data_processor_base():
    """Test the DataProcessor base class."""
    processor = MockProcessor()
    data = pd.DataFrame({"a": [1, 2, 3]})

    # Test the __call__ method
    result = processor(data)
    assert processor.processed
    assert isinstance(result, pd.DataFrame)
    assert result.equals(data)


def test_data_ingestion_pipeline(sample_data):
    """Test the DataIngestionPipeline class."""
    # Create a pipeline with multiple processors
    pipeline = DataIngestionPipeline(name="TestPipeline")

    processor1 = MockProcessor("Processor1")
    processor2 = MockProcessor("Processor2")
    processor3 = MockProcessor("Processor3")

    pipeline.add_processor(processor1)
    pipeline.add_processor(processor2)
    pipeline.add_processor(processor3)

    # Run the pipeline
    result = pipeline.run(sample_data)

    # Check that all processors were called
    assert processor1.processed
    assert processor2.processed
    assert processor3.processed

    # Result should be the same as input for mock processors
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_data)


def test_pipeline_error_handling(sample_data):
    """Test error handling in the pipeline."""
    pipeline = DataIngestionPipeline(name="ErrorPipeline")

    processor1 = MockProcessor("Processor1")
    processor2 = MockProcessor("Processor2", raise_error=True)
    processor3 = MockProcessor("Processor3")

    pipeline.add_processor(processor1)
    pipeline.add_processor(processor2)
    pipeline.add_processor(processor3)

    # Run the pipeline and expect an error
    with pytest.raises(ValueError, match="Mock processing error"):
        pipeline.run(sample_data)

    # First processor should have been called, but not the third
    assert processor1.processed
    assert not processor3.processed


def test_cleaning_processor(sample_data):
    """Test the CleaningProcessor."""
    processor = CleaningProcessor(
        drop_na_columns=["budget_usd", "box_office_revenue"],
        fill_na_values={"runtime_minutes": 120},
    )

    result = processor(sample_data)

    # Check that rows with NaN in critical columns were dropped
    assert len(result) < len(sample_data)
    assert len(result) == 3  # Two rows should be dropped

    # Check that column names were standardized (they should already be lowercase/snake_case)
    assert "budget_usd" in result.columns
    assert "box_office_revenue" in result.columns


def test_date_feature_processor(sample_data):
    """Test the DateFeatureProcessor."""
    processor = DateFeatureProcessor(
        date_columns=["release_date"], output_format="separate"
    )

    result = processor(sample_data)

    # Check that date features were created
    assert "release_date_year" in result.columns
    assert "release_date_month" in result.columns
    assert "release_date_quarter" in result.columns
    assert "release_date_is_weekend" in result.columns

    # Check some values
    assert result["release_date_year"].iloc[0] == 2020
    assert result["release_date_month"].iloc[0] == 1


def test_categorical_encoding_processor(sample_data):
    """Test the CategoricalEncodingProcessor."""
    processor = CategoricalEncodingProcessor(
        categorical_columns=["genre"], method="onehot"
    )

    result = processor(sample_data)

    # Check that one-hot encoded columns were created
    assert "genre_Action" in result.columns
    assert "genre_Comedy" in result.columns
    assert "genre_Drama" in result.columns
    assert "genre_Sci-Fi" in result.columns

    # Original column should be removed
    assert "genre" not in result.columns

    # Check some values
    assert result["genre_Action"].iloc[0] == 1
    assert result["genre_Comedy"].iloc[1] == 1


def test_numerical_feature_processor(sample_data):
    """Test the NumericalFeatureProcessor."""
    processor = NumericalFeatureProcessor(
        numerical_columns=["budget_usd", "runtime_minutes"],
        transformations={"log": ["budget_usd"], "minmax": ["runtime_minutes"]},
    )

    result = processor(sample_data)

    # Check that transformed columns were created
    assert "budget_usd_log" in result.columns
    assert "runtime_minutes_minmax" in result.columns

    # Check some values
    assert abs(result["budget_usd_log"].iloc[0] - np.log(100000)) < 0.0001
    assert 0.0 <= result["runtime_minutes_minmax"].iloc[0] <= 1.0


def test_pipeline_save_results(sample_data):
    """Test saving results from the pipeline."""
    pipeline = DataIngestionPipeline(name="SavePipeline")

    processor = CleaningProcessor(drop_na_columns=["budget_usd", "box_office_revenue"])

    pipeline.add_processor(processor)

    # Run the pipeline
    result = pipeline.run(sample_data)

    # Save the results to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        output_path = tmp.name

    try:
        pipeline.save_processed_data(result, output_path)

        # Check that the file was created and has the correct data
        assert os.path.exists(output_path)
        loaded_data = pd.read_csv(output_path)
        assert len(loaded_data) == len(result)
        assert set(loaded_data.columns) == set(result.columns)

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
