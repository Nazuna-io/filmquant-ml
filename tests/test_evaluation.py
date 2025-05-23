# tests/test_evaluation.py
"""
Tests for the model evaluation framework.
"""
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import (
    calculate_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    quantile_loss,
    root_mean_squared_error,
)
from app.evaluation.model_tracker import ModelTracker


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for testing."""
    np.random.seed(42)
    y_true = np.random.normal(loc=100, scale=20, size=100)
    y_pred = y_true + np.random.normal(loc=0, scale=10, size=100)
    y_pred_low = y_pred - np.abs(np.random.normal(loc=0, scale=15, size=100))
    y_pred_high = y_pred + np.abs(np.random.normal(loc=0, scale=15, size=100))

    return y_true, y_pred, y_pred_low, y_pred_high


def test_mean_absolute_error(sample_predictions):
    """Test the mean_absolute_error function."""
    y_true, y_pred, _, _ = sample_predictions

    mae = mean_absolute_error(y_true, y_pred)

    # MAE should be positive
    assert mae >= 0

    # MAE should be correct
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert mae == expected_mae


def test_mean_absolute_percentage_error(sample_predictions):
    """Test the mean_absolute_percentage_error function."""
    y_true, y_pred, _, _ = sample_predictions

    mape = mean_absolute_percentage_error(y_true, y_pred)

    # MAPE should be positive
    assert mape >= 0

    # MAPE should be correct (as a percentage)
    expected_mape = 100.0 * np.mean(np.abs((y_true - y_pred) / y_true))
    assert abs(mape - expected_mape) < 0.0001


def test_root_mean_squared_error(sample_predictions):
    """Test the root_mean_squared_error function."""
    y_true, y_pred, _, _ = sample_predictions

    rmse = root_mean_squared_error(y_true, y_pred)

    # RMSE should be positive
    assert rmse >= 0

    # RMSE should be correct
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert abs(rmse - expected_rmse) < 0.0001


def test_quantile_loss(sample_predictions):
    """Test the quantile_loss function."""
    y_true, _, y_pred_low, y_pred_high = sample_predictions

    ql = quantile_loss(y_true, y_pred_low, y_pred_high)

    # Should return a dictionary with expected keys
    assert isinstance(ql, dict)
    assert "lower_quantile_loss" in ql
    assert "upper_quantile_loss" in ql
    assert "interval_width" in ql
    assert "interval_coverage_percent" in ql

    # Coverage should be between 0 and 100
    assert 0 <= ql["interval_coverage_percent"] <= 100


def test_calculate_metrics(sample_predictions):
    """Test the calculate_metrics function."""
    y_true, y_pred, y_pred_low, y_pred_high = sample_predictions

    # Test with just point predictions
    metrics = calculate_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "mape" in metrics
    assert "rmse" in metrics

    # Test with prediction intervals
    metrics_with_intervals = calculate_metrics(y_true, y_pred, y_pred_low, y_pred_high)

    assert "mae" in metrics_with_intervals
    assert "mape" in metrics_with_intervals
    assert "rmse" in metrics_with_intervals
    assert "interval_coverage_percent" in metrics_with_intervals


def test_model_tracker_initialization():
    """Test initialization of the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Check that directories were created
        assert os.path.exists(os.path.join(temp_dir, "test_experiment"))
        assert os.path.exists(os.path.join(temp_dir, "test_experiment", "models"))
        assert os.path.exists(os.path.join(temp_dir, "test_experiment", "metrics"))
        assert os.path.exists(os.path.join(temp_dir, "test_experiment", "plots"))


def test_model_tracker_log_hyperparameters():
    """Test logging hyperparameters with the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Log hyperparameters
        hp = {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 3}
        hp_file = tracker.log_hyperparameters("model_v1", hp)

        # Check that file was created
        assert os.path.exists(hp_file)

        # Check file contents
        with open(hp_file, "r") as f:
            loaded_hp = json.load(f)

        assert loaded_hp["learning_rate"] == hp["learning_rate"]
        assert loaded_hp["n_estimators"] == hp["n_estimators"]
        assert loaded_hp["max_depth"] == hp["max_depth"]
        assert "timestamp" in loaded_hp
        assert loaded_hp["model_version"] == "model_v1"


def test_model_tracker_log_model_files():
    """Test logging model files with the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Create temporary model files
        model_file = tempfile.NamedTemporaryFile(delete=False)
        processor_file = tempfile.NamedTemporaryFile(delete=False)

        try:
            # Write some content to the files
            with open(model_file.name, "w") as f:
                f.write("model content")
            with open(processor_file.name, "w") as f:
                f.write("processor content")

            # Log model files
            model_files = {"model": model_file.name, "processor": processor_file.name}
            tracked_files = tracker.log_model_files("model_v1", model_files)

            # Check that files were copied
            assert os.path.exists(tracked_files["model"])
            assert os.path.exists(tracked_files["processor"])

            # Check file contents
            with open(tracked_files["model"], "r") as f:
                assert f.read() == "model content"
            with open(tracked_files["processor"], "r") as f:
                assert f.read() == "processor content"

        finally:
            # Clean up
            os.unlink(model_file.name)
            os.unlink(processor_file.name)


def test_model_tracker_evaluate_model(sample_predictions):
    """Test evaluating a model with the ModelTracker."""
    y_true, y_pred, y_pred_low, y_pred_high = sample_predictions

    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Create feature importance dictionary
        feature_importance = {"feature1": 100.0, "feature2": 80.0, "feature3": 60.0}

        # Evaluate model
        metrics = tracker.evaluate_model(
            model_version="model_v1",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_low=y_pred_low,
            y_pred_high=y_pred_high,
            feature_importance=feature_importance,
            generate_plots=True,
        )

        # Check that metrics were calculated
        assert "mae" in metrics
        assert "mape" in metrics
        assert "rmse" in metrics
        assert "interval_coverage_percent" in metrics

        # Check that metrics file was created
        metrics_file = os.path.join(
            temp_dir, "test_experiment", "metrics", "model_v1_metrics.json"
        )
        assert os.path.exists(metrics_file)

        # Check that plots were generated
        plots_dir = os.path.join(temp_dir, "test_experiment", "plots", "model_v1")
        assert os.path.exists(plots_dir)
        assert os.path.exists(os.path.join(plots_dir, "actual_vs_predicted.png"))
        assert os.path.exists(os.path.join(plots_dir, "error_distribution.png"))
        assert os.path.exists(os.path.join(plots_dir, "residuals_vs_predicted.png"))
        assert os.path.exists(os.path.join(plots_dir, "feature_importance.png"))


def test_model_tracker_compare_models():
    """Test comparing models with the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Create metrics for two models
        metrics1 = {
            "model_version": "model_v1",
            "mae": 10.0,
            "mape": 5.0,
            "rmse": 12.0,
            "interval_coverage_percent": 90.0,
        }

        metrics2 = {
            "model_version": "model_v2",
            "mae": 8.0,
            "mape": 4.0,
            "rmse": 10.0,
            "interval_coverage_percent": 92.0,
        }

        # Save metrics
        metrics_dir = os.path.join(temp_dir, "test_experiment", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        with open(os.path.join(metrics_dir, "model_v1_metrics.json"), "w") as f:
            json.dump(metrics1, f)

        with open(os.path.join(metrics_dir, "model_v2_metrics.json"), "w") as f:
            json.dump(metrics2, f)

        # Compare models
        comparison = tracker.compare_models(["model_v1", "model_v2"])

        # Check comparison
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert set(comparison["model_version"]) == {"model_v1", "model_v2"}

        # v2 should have better metrics
        assert (
            comparison.loc[comparison["model_version"] == "model_v2", "mae"].iloc[0]
            < comparison.loc[comparison["model_version"] == "model_v1", "mae"].iloc[0]
        )


def test_model_tracker_generate_report():
    """Test generating a report with the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Create metrics
        metrics = {
            "model_version": "model_v1",
            "mae": 10.0,
            "mape": 5.0,
            "rmse": 12.0,
            "interval_coverage_percent": 90.0,
            "feature_importance": [
                {"feature": "feature1", "importance": 100.0},
                {"feature": "feature2", "importance": 80.0},
                {"feature": "feature3", "importance": 60.0},
            ],
        }

        # Save metrics
        metrics_dir = os.path.join(temp_dir, "test_experiment", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        with open(os.path.join(metrics_dir, "model_v1_metrics.json"), "w") as f:
            json.dump(metrics, f)

        # Create hyperparameters
        hp = {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 3}

        # Save hyperparameters
        model_dir = os.path.join(temp_dir, "test_experiment", "models", "model_v1")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

        # Generate report
        report = tracker.generate_report("model_v1", output_format="json")

        # Check report
        assert report["model_version"] == "model_v1"
        assert "mae" in report
        assert "hyperparameters" in report

        # Generate HTML report
        html_report = tracker.generate_report("model_v1", output_format="html")

        # Check HTML report
        html_file = os.path.join(temp_dir, "test_experiment", "model_v1_report.html")
        assert os.path.exists(html_file)


def test_model_tracker_get_best_model():
    """Test getting the best model with the ModelTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ModelTracker(temp_dir, "test_experiment")

        # Create metrics for three models
        metrics1 = {"model_version": "model_v1", "mae": 10.0, "mape": 5.0, "rmse": 12.0}

        metrics2 = {"model_version": "model_v2", "mae": 8.0, "mape": 4.0, "rmse": 10.0}

        metrics3 = {"model_version": "model_v3", "mae": 9.0, "mape": 4.5, "rmse": 11.0}

        # Save metrics
        metrics_dir = os.path.join(temp_dir, "test_experiment", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        with open(os.path.join(metrics_dir, "model_v1_metrics.json"), "w") as f:
            json.dump(metrics1, f)

        with open(os.path.join(metrics_dir, "model_v2_metrics.json"), "w") as f:
            json.dump(metrics2, f)

        with open(os.path.join(metrics_dir, "model_v3_metrics.json"), "w") as f:
            json.dump(metrics3, f)

        # Create model directories
        for model in ["model_v1", "model_v2", "model_v3"]:
            os.makedirs(
                os.path.join(temp_dir, "test_experiment", "models", model),
                exist_ok=True,
            )

        # Get best model by MAE (lower is better)
        best_model = tracker.get_best_model_version(
            metric="mae", higher_is_better=False
        )
        assert best_model == "model_v2"

        # Get best model by MAE (pretending higher is better)
        best_model = tracker.get_best_model_version(metric="mae", higher_is_better=True)
        assert best_model == "model_v1"
