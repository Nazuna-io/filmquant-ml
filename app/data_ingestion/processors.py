# app/data_ingestion/processors.py
"""
Specific data processors for the data ingestion pipeline.
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.data_ingestion.pipeline import DataProcessor
from app.utils.logging import get_logger

logger = get_logger("filmquant_ml.data_ingestion.processors")


class CleaningProcessor(DataProcessor):
    """
    Processor for data cleaning operations including:
    - Handling missing values
    - Removing duplicates
    - Standardizing column names
    """

    def __init__(
        self, drop_na_columns: List[str] = None, fill_na_values: Dict[str, Any] = None
    ):
        """
        Initialize the cleaning processor.

        Args:
            drop_na_columns: Columns where rows with NaN values should be dropped
            fill_na_values: Dictionary mapping column names to values for filling NaN values
        """
        super().__init__(name="CleaningProcessor")
        self.drop_na_columns = drop_na_columns or []
        self.fill_na_values = fill_na_values or {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data with cleaning operations.

        Args:
            data: Input DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = data.copy()

        # Standardize column names (lowercase, snake_case)
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        self.logger.debug(f"Standardized column names: {list(df.columns)}")

        # Drop duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_dups = initial_rows - len(df)
        self.logger.info(f"Dropped {dropped_dups} duplicate rows")

        # Handle missing values - first drop rows with NaN in critical columns
        if self.drop_na_columns:
            initial_rows = len(df)
            df = df.dropna(subset=self.drop_na_columns)
            dropped_rows = initial_rows - len(df)
            self.logger.info(
                f"Dropped {dropped_rows} rows with NaN values in columns: {self.drop_na_columns}"
            )

        # Fill remaining NaN values
        for column, fill_value in self.fill_na_values.items():
            if column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    df[column] = df[column].fillna(fill_value)
                    self.logger.info(
                        f"Filled {missing_count} NaN values in '{column}' with '{fill_value}'"
                    )

        return df

    def _standardize_column_name(self, column_name: str) -> str:
        """
        Standardize a column name to snake_case.

        Args:
            column_name: Original column name

        Returns:
            Standardized column name
        """
        # Replace spaces and special characters with underscores
        name = re.sub(r"[^a-zA-Z0-9]", "_", column_name)
        # Convert to lowercase
        name = name.lower()
        # Replace multiple consecutive underscores with a single one
        name = re.sub(r"_+", "_", name)
        # Remove leading and trailing underscores
        name = name.strip("_")
        return name


class DateFeatureProcessor(DataProcessor):
    """
    Processor for extracting features from date columns, such as:
    - Month, quarter, year
    - Season
    - Is holiday/weekend
    """

    def __init__(self, date_columns: List[str], output_format: str = "separate"):
        """
        Initialize the date feature processor.

        Args:
            date_columns: List of column names containing date information
            output_format: Format for extracted features:
                - 'separate': Create separate columns for each feature
                - 'onehot': One-hot encode categorical features
        """
        super().__init__(name="DateFeatureProcessor")
        self.date_columns = date_columns
        self.output_format = output_format

        # Define US holidays (simplified for demonstration)
        self.us_holidays = {
            # Format: 'MM-DD': 'Holiday Name'
            "01-01": "New Year",
            "07-04": "Independence Day",
            "12-25": "Christmas",
            # Add more holidays as needed
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data to extract date features.

        Args:
            data: Input DataFrame with date columns

        Returns:
            DataFrame with extracted date features
        """
        df = data.copy()

        for date_col in self.date_columns:
            if date_col not in df.columns:
                self.logger.warning(f"Date column '{date_col}' not found in DataFrame")
                continue

            # Convert column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    self.logger.debug(f"Converted '{date_col}' to datetime")
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert '{date_col}' to datetime: {str(e)}"
                    )
                    continue

            # Extract date features
            col_prefix = f"{date_col}_"

            # Basic date components
            df[f"{col_prefix}year"] = df[date_col].dt.year
            df[f"{col_prefix}month"] = df[date_col].dt.month
            df[f"{col_prefix}day"] = df[date_col].dt.day
            df[f"{col_prefix}dayofweek"] = df[date_col].dt.dayofweek

            # Quarter
            df[f"{col_prefix}quarter"] = df[date_col].dt.quarter

            # Is weekend
            df[f"{col_prefix}is_weekend"] = df[date_col].dt.dayofweek >= 5

            # Season (Northern Hemisphere)
            # 1: Winter (Dec-Feb), 2: Spring (Mar-May), 3: Summer (Jun-Aug), 4: Fall (Sep-Nov)
            month = df[date_col].dt.month
            df[f"{col_prefix}season"] = np.select(
                [
                    (month == 12) | (month <= 2),
                    (month >= 3) & (month <= 5),
                    (month >= 6) & (month <= 8),
                    (month >= 9) & (month <= 11),
                ],
                [1, 2, 3, 4],
            )

            # Is holiday
            month_day = df[date_col].dt.strftime("%m-%d")
            df[f"{col_prefix}is_holiday"] = month_day.isin(self.us_holidays.keys())

            # One-hot encode categorical features if specified
            if self.output_format == "onehot":
                # Month one-hot
                month_dummies = pd.get_dummies(
                    df[f"{col_prefix}month"], prefix=f"{col_prefix}month"
                )
                df = pd.concat([df, month_dummies], axis=1)

                # Quarter one-hot
                quarter_dummies = pd.get_dummies(
                    df[f"{col_prefix}quarter"], prefix=f"{col_prefix}quarter"
                )
                df = pd.concat([df, quarter_dummies], axis=1)

                # Season one-hot
                season_dummies = pd.get_dummies(
                    df[f"{col_prefix}season"], prefix=f"{col_prefix}season"
                )
                df = pd.concat([df, season_dummies], axis=1)

        return df


class CategoricalEncodingProcessor(DataProcessor):
    """
    Processor for encoding categorical variables using various methods:
    - One-hot encoding
    - Label encoding
    - Target encoding
    """

    def __init__(
        self,
        categorical_columns: List[str],
        method: str = "onehot",
        target_column: str = None,
    ):
        """
        Initialize the categorical encoding processor.

        Args:
            categorical_columns: List of column names containing categorical variables
            method: Encoding method ('onehot', 'label', 'target')
            target_column: Target column for target encoding (required if method='target')
        """
        super().__init__(name="CategoricalEncodingProcessor")
        self.categorical_columns = categorical_columns
        self.method = method
        self.target_column = target_column

        if method == "target" and not target_column:
            raise ValueError("target_column is required for target encoding")

        # Store encoders for future transformations (e.g. when applying to new data)
        self.encoders = {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data to encode categorical variables.

        Args:
            data: Input DataFrame with categorical columns

        Returns:
            DataFrame with encoded categorical variables
        """
        df = data.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                self.logger.warning(
                    f"Categorical column '{col}' not found in DataFrame"
                )
                continue

            if self.method == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                self.logger.debug(
                    f"One-hot encoded '{col}' into {dummies.shape[1]} columns"
                )

            elif self.method == "label":
                # Label encoding (simple)
                categories = df[col].unique()
                category_map = {cat: i for i, cat in enumerate(categories)}
                df[f"{col}_encoded"] = df[col].map(category_map)
                self.encoders[col] = category_map
                self.logger.debug(
                    f"Label encoded '{col}' with {len(category_map)} categories"
                )

            elif self.method == "target":
                # Target encoding
                target_means = df.groupby(col)[self.target_column].mean()
                df[f"{col}_target_encoded"] = df[col].map(target_means)
                self.encoders[col] = target_means.to_dict()
                self.logger.debug(
                    f"Target encoded '{col}' using '{self.target_column}'"
                )

        return df


class NumericalFeatureProcessor(DataProcessor):
    """
    Processor for transforming numerical features:
    - Scaling (min-max, standard, robust)
    - Binning
    - Log transformation
    - Polynomial features
    """

    def __init__(
        self, numerical_columns: List[str], transformations: Dict[str, List[str]]
    ):
        """
        Initialize the numerical feature processor.

        Args:
            numerical_columns: List of column names containing numerical variables
            transformations: Dictionary mapping transformation types to columns
                e.g. {'log': ['budget', 'revenue'], 'bins': ['runtime']}
        """
        super().__init__(name="NumericalFeatureProcessor")
        self.numerical_columns = numerical_columns
        self.transformations = transformations

        # Store scalers and transformers for future use
        self.scaling_params = {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data to transform numerical features.

        Args:
            data: Input DataFrame with numerical columns

        Returns:
            DataFrame with transformed numerical features
        """
        df = data.copy()

        # Apply log transformation
        if "log" in self.transformations:
            for col in self.transformations["log"]:
                if col not in df.columns:
                    self.logger.warning(
                        f"Column '{col}' not found for log transformation"
                    )
                    continue

                # Check if column contains positive values only
                if (df[col] <= 0).any():
                    log_col = f"{col}_log1p"
                    df[log_col] = np.log1p(df[col])
                    self.logger.debug(
                        f"Applied log1p transformation to '{col}' -> '{log_col}'"
                    )
                else:
                    log_col = f"{col}_log"
                    df[log_col] = np.log(df[col])
                    self.logger.debug(
                        f"Applied log transformation to '{col}' -> '{log_col}'"
                    )

        # Apply min-max scaling
        if "minmax" in self.transformations:
            for col in self.transformations["minmax"]:
                if col not in df.columns:
                    self.logger.warning(f"Column '{col}' not found for min-max scaling")
                    continue

                col_min = df[col].min()
                col_max = df[col].max()
                scaled_col = f"{col}_minmax"
                df[scaled_col] = (df[col] - col_min) / (col_max - col_min)

                self.scaling_params[col] = {"min": col_min, "max": col_max}
                self.logger.debug(
                    f"Applied min-max scaling to '{col}' -> '{scaled_col}'"
                )

        # Apply binning
        if "bins" in self.transformations:
            for col in self.transformations["bins"]:
                if col not in df.columns:
                    self.logger.warning(f"Column '{col}' not found for binning")
                    continue

                # Determine number of bins based on data distribution
                n_bins = min(10, len(df[col].unique()))
                if n_bins < 2:
                    self.logger.warning(
                        f"Not enough unique values in '{col}' for binning"
                    )
                    continue

                # Create bins
                binned_col = f"{col}_binned"
                df[binned_col] = pd.qcut(
                    df[col], q=n_bins, duplicates="drop", labels=False
                )
                self.logger.debug(
                    f"Applied binning to '{col}' -> '{binned_col}' with {n_bins} bins"
                )

        # Apply standard scaling
        if "standard" in self.transformations:
            for col in self.transformations["standard"]:
                if col not in df.columns:
                    self.logger.warning(
                        f"Column '{col}' not found for standard scaling"
                    )
                    continue

                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std == 0:
                    self.logger.warning(
                        f"Standard deviation is 0 for '{col}', skipping scaling"
                    )
                    continue

                scaled_col = f"{col}_scaled"
                df[scaled_col] = (df[col] - col_mean) / col_std

                self.scaling_params[col] = {"mean": col_mean, "std": col_std}
                self.logger.debug(
                    f"Applied standard scaling to '{col}' -> '{scaled_col}'"
                )

        return df


class DataSplitProcessor(DataProcessor):
    """
    Processor for splitting data into training and validation sets.
    """

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        split_method: str = "random",
        split_column: str = None,
    ):
        """
        Initialize the data split processor.

        Args:
            target_column: Name of the target column
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            split_method: Method for splitting ('random', 'time', 'stratified')
            split_column: Column to use for non-random splits (e.g. date column for time-based splits)
        """
        super().__init__(name="DataSplitProcessor")
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.split_method = split_method
        self.split_column = split_column

        if split_method in ["time", "stratified"] and not split_column:
            raise ValueError(f"split_column is required for {split_method} splitting")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data to split into training and validation sets.

        Args:
            data: Input DataFrame to split

        Returns:
            DataFrame with an additional column 'split' indicating the split assignment
        """
        df = data.copy()

        # Make sure target column exists
        if self.target_column not in df.columns:
            self.logger.error(
                f"Target column '{self.target_column}' not found in DataFrame"
            )
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Create a split column
        df["split"] = "train"  # Default all to training

        if self.split_method == "random":
            # Random split
            np.random.seed(self.random_state)
            mask = np.random.rand(len(df)) < self.test_size
            df.loc[mask, "split"] = "test"
            self.logger.info(
                f"Randomly split data: {(~mask).sum()} train, {mask.sum()} test samples"
            )

        elif self.split_method == "time":
            # Time-based split (assumes split_column is sorted chronologically)
            if self.split_column not in df.columns:
                self.logger.error(
                    f"Split column '{self.split_column}' not found for time-based splitting"
                )
                raise ValueError(f"Split column '{self.split_column}' not found")

            # Sort by time
            df = df.sort_values(by=self.split_column)

            # Select the last test_size fraction for testing
            test_count = int(len(df) * self.test_size)
            df.iloc[-test_count:, df.columns.get_loc("split")] = "test"
            self.logger.info(
                f"Time-split data: {len(df) - test_count} train, {test_count} test samples"
            )

        elif self.split_method == "stratified":
            # Stratified split (ensures same distribution of split_column in train/test)
            if self.split_column not in df.columns:
                self.logger.error(
                    f"Split column '{self.split_column}' not found for stratified splitting"
                )
                raise ValueError(f"Split column '{self.split_column}' not found")

            # Get unique values in split column
            strata = df[self.split_column].unique()

            # For each stratum, select test_size fraction for testing
            for stratum in strata:
                stratum_indices = df[df[self.split_column] == stratum].index
                test_count = int(len(stratum_indices) * self.test_size)

                # Randomly select indices for test set
                np.random.seed(self.random_state)
                test_indices = np.random.choice(
                    stratum_indices, test_count, replace=False
                )
                df.loc[test_indices, "split"] = "test"

            test_count = (df["split"] == "test").sum()
            self.logger.info(
                f"Stratified-split data: {len(df) - test_count} train, {test_count} test samples"
            )

        return df


class FeatureSelectionProcessor(DataProcessor):
    """
    Processor for feature selection to reduce dimensionality.
    """

    def __init__(
        self,
        target_column: str,
        method: str = "correlation",
        threshold: float = 0.05,
        max_features: int = None,
        include_columns: List[str] = None,
    ):
        """
        Initialize the feature selection processor.

        Args:
            target_column: Name of the target column
            method: Feature selection method ('correlation', 'variance', 'importance')
            threshold: Threshold for feature selection
            max_features: Maximum number of features to select
            include_columns: Columns to always include regardless of selection criteria
        """
        super().__init__(name="FeatureSelectionProcessor")
        self.target_column = target_column
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self.include_columns = include_columns or []

        # Store selected features for future use
        self.selected_features = None

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data to select relevant features.

        Args:
            data: Input DataFrame with features and target

        Returns:
            DataFrame with selected features
        """
        df = data.copy()

        # Make sure target column exists
        if self.target_column not in df.columns:
            self.logger.error(
                f"Target column '{self.target_column}' not found in DataFrame"
            )
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Get numerical columns (potential features)
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numerical_cols = [
            col
            for col in numerical_cols
            if col != self.target_column and col != "split"
        ]

        if self.method == "correlation":
            # Calculate correlation with target
            correlations = (
                df[numerical_cols + [self.target_column]]
                .corr()[self.target_column]
                .abs()
            )
            correlations = correlations.drop(self.target_column)

            # Select features based on correlation threshold
            selected = correlations[correlations > self.threshold].index.tolist()
            self.logger.info(
                f"Selected {len(selected)} features based on correlation > {self.threshold}"
            )

            # If max_features is set, take top N features
            if self.max_features and len(selected) > self.max_features:
                selected = (
                    correlations.sort_values(ascending=False)
                    .head(self.max_features)
                    .index.tolist()
                )
                self.logger.info(
                    f"Reduced to top {self.max_features} features by correlation"
                )

            # Include required columns
            selected = list(set(selected + self.include_columns))

        elif self.method == "variance":
            # Calculate variance of features
            variances = df[numerical_cols].var()

            # Select features based on variance threshold
            selected = variances[variances > self.threshold].index.tolist()
            self.logger.info(
                f"Selected {len(selected)} features based on variance > {self.threshold}"
            )

            # If max_features is set, take top N features
            if self.max_features and len(selected) > self.max_features:
                selected = (
                    variances.sort_values(ascending=False)
                    .head(self.max_features)
                    .index.tolist()
                )
                self.logger.info(
                    f"Reduced to top {self.max_features} features by variance"
                )

            # Include required columns
            selected = list(set(selected + self.include_columns))

        else:
            # Default: include all numerical columns and required columns
            selected = list(set(numerical_cols + self.include_columns))
            self.logger.info(
                f"Using all {len(selected)} numerical features (no selection applied)"
            )

        # Always include target column and split column if present
        if "split" in df.columns:
            selected = list(set(selected + [self.target_column, "split"]))
        else:
            selected = list(set(selected + [self.target_column]))

        # Store selected features for future use
        self.selected_features = selected

        # Return DataFrame with selected columns
        return df[selected]
