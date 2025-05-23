# app/evaluation/evaluate.py
"""
Evaluation script for FilmQuant ML.
"""
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.evaluation.metrics import calculate_metrics, evaluate_stratified
from app.evaluation.model_tracker import ModelTracker
from app.evaluation.visualization import (
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_residuals_vs_predicted,
    plot_stratified_metrics,
)
from app.utils.logging import configure_logging, get_logger

# Configure logging
logger = configure_logging(app_name="filmquant_ml.evaluation", log_level=logging.INFO)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load evaluation data from a CSV file.

    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with evaluation data
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {str(e)}")
        raise


def evaluate_model(
    data: pd.DataFrame,
    model_version: str,
    true_column: str,
    pred_column: str,
    pred_low_column: Optional[str] = None,
    pred_high_column: Optional[str] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    output_dir: str = "evaluation_results",
    stratify_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on the given data.

    Args:
        data: DataFrame with true and predicted values
        model_version: Version identifier for the model
        true_column: Name of the column with true values
        pred_column: Name of the column with predicted values
        pred_low_column: Name of the column with lower bound predictions (optional)
        pred_high_column: Name of the column with upper bound predictions (optional)
        feature_importance: Dictionary mapping feature names to importance scores (optional)
        output_dir: Directory to save evaluation results
        stratify_columns: List of columns to use for stratified evaluation (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check if required columns exist
    if true_column not in data.columns:
        logger.error(f"True value column '{true_column}' not found in data")
        raise ValueError(f"True value column '{true_column}' not found in data")

    if pred_column not in data.columns:
        logger.error(f"Predicted value column '{pred_column}' not found in data")
        raise ValueError(f"Predicted value column '{pred_column}' not found in data")

    # Get prediction intervals if available
    y_pred_low = None
    y_pred_high = None

    if pred_low_column and pred_high_column:
        if pred_low_column in data.columns and pred_high_column in data.columns:
            y_pred_low = data[pred_low_column].values
            y_pred_high = data[pred_high_column].values
            logger.info("Using prediction intervals for evaluation")
        else:
            logger.warning("Prediction interval columns not found in data")

    # Extract true and predicted values
    y_true = data[true_column].values
    y_pred = data[pred_column].values

    # Calculate metrics
    logger.info("Calculating evaluation metrics")
    metrics = calculate_metrics(y_true, y_pred, y_pred_low, y_pred_high)

    # Add model version and timestamp
    metrics["model_version"] = model_version
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["sample_size"] = len(data)

    # Add feature importance if provided
    if feature_importance:
        metrics["feature_importance"] = {
            k: float(v) for k, v in feature_importance.items()
        }

    # Calculate stratified metrics if requested
    if stratify_columns:
        stratified_metrics = {}
        for column in stratify_columns:
            if column in data.columns:
                logger.info(f"Calculating stratified metrics for '{column}'")
                strata = {}
                for value in data[column].unique():
                    stratum_mask = data[column] == value
                    strata[f"{column}_{value}"] = stratum_mask.values

                column_metrics = evaluate_stratified(y_true, y_pred, strata)
                stratified_metrics[column] = column_metrics
            else:
                logger.warning(f"Stratification column '{column}' not found in data")

        if stratified_metrics:
            metrics["stratified_metrics"] = stratified_metrics

    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, f"{model_version}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Generate plots
    logger.info("Generating evaluation plots")

    # Actual vs Predicted plot
    plot_file = os.path.join(plots_dir, f"{model_version}_actual_vs_predicted.png")
    plot_actual_vs_predicted(y_true, y_pred, y_pred_low, y_pred_high, plot_file)

    # Error distribution plot
    plot_file = os.path.join(plots_dir, f"{model_version}_error_distribution.png")
    plot_error_distribution(y_true, y_pred, plot_file)

    # Residuals vs Predicted plot
    plot_file = os.path.join(plots_dir, f"{model_version}_residuals.png")
    plot_residuals_vs_predicted(y_true, y_pred, plot_file)

    # Stratified metrics plots
    if "stratified_metrics" in metrics:
        for column, strata_metrics in metrics["stratified_metrics"].items():
            plot_file = os.path.join(
                plots_dir, f"{model_version}_stratified_{column}.png"
            )
            plot_stratified_metrics(
                strata_metrics, "mae", f"MAE by {column}", plot_file
            )

    logger.info("Evaluation completed successfully")
    return metrics


def main(
    data_path: str,
    model_version: str,
    true_column: str,
    pred_column: str,
    pred_low_column: Optional[str] = None,
    pred_high_column: Optional[str] = None,
    experiment_name: Optional[str] = None,
    feature_importance_file: Optional[str] = None,
    output_dir: str = "evaluation_results",
    stratify_columns: Optional[List[str]] = None,
):
    """
    Main function for model evaluation.

    Args:
        data_path: Path to the data CSV file
        model_version: Version identifier for the model
        true_column: Name of the column with true values
        pred_column: Name of the column with predicted values
        pred_low_column: Name of the column with lower bound predictions (optional)
        pred_high_column: Name of the column with upper bound predictions (optional)
        experiment_name: Name of the experiment for the model tracker (optional)
        feature_importance_file: Path to a JSON file with feature importance scores (optional)
        output_dir: Directory to save evaluation results
        stratify_columns: List of columns to use for stratified evaluation (optional)
    """
    logger.info(f"Starting evaluation for model version {model_version}")

    try:
        # Load data
        data = load_data(data_path)

        # Load feature importance if provided
        feature_importance = None
        if feature_importance_file and os.path.exists(feature_importance_file):
            try:
                with open(feature_importance_file, "r") as f:
                    feature_importance = json.load(f)
                logger.info(f"Loaded feature importance from {feature_importance_file}")
            except Exception as e:
                logger.warning(f"Failed to load feature importance: {str(e)}")

        # Configure model tracker if experiment name is provided
        if experiment_name:
            tracker_dir = os.path.join(output_dir, "model_tracking")
            tracker = ModelTracker(tracker_dir, experiment_name)

            # Evaluate with model tracker
            strata = {}
            if stratify_columns:
                for column in stratify_columns:
                    if column in data.columns:
                        for value in data[column].unique():
                            stratum_mask = data[column] == value
                            strata[f"{column}_{value}"] = stratum_mask.values

            tracker.evaluate_model(
                model_version=model_version,
                y_true=data[true_column].values,
                y_pred=data[pred_column].values,
                y_pred_low=(
                    data[pred_low_column].values
                    if pred_low_column in data.columns
                    else None
                ),
                y_pred_high=(
                    data[pred_high_column].values
                    if pred_high_column in data.columns
                    else None
                ),
                feature_importance=feature_importance,
                strata=strata if strata else None,
                generate_plots=True,
            )

            # Generate HTML report
            tracker.generate_report(model_version, "html")
            logger.info(
                f"Generated evaluation report with model tracker for {model_version}"
            )
        else:
            # Evaluate without model tracker
            evaluate_model(
                data=data,
                model_version=model_version,
                true_column=true_column,
                pred_column=pred_column,
                pred_low_column=pred_low_column,
                pred_high_column=pred_high_column,
                feature_importance=feature_importance,
                output_dir=output_dir,
                stratify_columns=stratify_columns,
            )

        logger.info("Evaluation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilmQuant ML Model Evaluation")
    parser.add_argument(
        "--data", "-d", required=True, help="Path to the evaluation data CSV file"
    )
    parser.add_argument(
        "--model-version",
        "-m",
        required=True,
        help="Version identifier for the model being evaluated",
    )
    parser.add_argument(
        "--true-column", "-t", required=True, help="Name of the column with true values"
    )
    parser.add_argument(
        "--pred-column",
        "-p",
        required=True,
        help="Name of the column with predicted values",
    )
    parser.add_argument(
        "--pred-low-column",
        "-l",
        default=None,
        help="Name of the column with lower bound predictions",
    )
    parser.add_argument(
        "--pred-high-column",
        "-u",
        default=None,
        help="Name of the column with upper bound predictions",
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        default=None,
        help="Name of the experiment for the model tracker",
    )
    parser.add_argument(
        "--feature-importance",
        "-f",
        default=None,
        help="Path to a JSON file with feature importance scores",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--stratify",
        "-s",
        nargs="+",
        default=None,
        help="List of columns to use for stratified evaluation",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)

    success = main(
        data_path=args.data,
        model_version=args.model_version,
        true_column=args.true_column,
        pred_column=args.pred_column,
        pred_low_column=args.pred_low_column,
        pred_high_column=args.pred_high_column,
        experiment_name=args.experiment_name,
        feature_importance_file=args.feature_importance,
        output_dir=args.output_dir,
        stratify_columns=args.stratify,
    )

    exit(0 if success else 1)
