# app/evaluation/metrics.py
"""
Metrics module for evaluating model performance.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

from app.utils.logging import get_logger

logger = get_logger("filmquant_ml.evaluation.metrics")

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        MAPE value as a percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def quantile_loss(y_true: np.ndarray, y_pred_low: np.ndarray, y_pred_high: np.ndarray) -> Dict[str, float]:
    """
    Calculate quantile loss for prediction intervals.
    
    Args:
        y_true: Array of true values
        y_pred_low: Array of lower bound predictions
        y_pred_high: Array of upper bound predictions
        
    Returns:
        Dictionary with lower quantile loss, upper quantile loss, and interval width metrics
    """
    # Calculate lower quantile loss (for q=0.05 typically)
    lower_q = 0.05
    lower_loss = np.mean(np.maximum(lower_q * (y_true - y_pred_low), (lower_q - 1) * (y_true - y_pred_low)))
    
    # Calculate upper quantile loss (for q=0.95 typically)
    upper_q = 0.95
    upper_loss = np.mean(np.maximum(upper_q * (y_true - y_pred_high), (upper_q - 1) * (y_true - y_pred_high)))
    
    # Calculate mean interval width as a percentage of the median prediction
    interval_width = np.mean((y_pred_high - y_pred_low))
    
    # Calculate interval coverage (percentage of true values within the prediction interval)
    in_interval = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    coverage = 100.0 * np.mean(in_interval)
    
    return {
        'lower_quantile_loss': float(lower_loss),
        'upper_quantile_loss': float(upper_loss),
        'interval_width': float(interval_width),
        'interval_coverage_percent': float(coverage)
    }

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_low: np.ndarray = None, 
                     y_pred_high: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate a suite of evaluation metrics.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values (median/mean predictions)
        y_pred_low: Array of lower bound predictions (optional)
        y_pred_high: Array of upper bound predictions (optional)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred)
    }
    
    # Add quantile metrics if prediction intervals are provided
    if y_pred_low is not None and y_pred_high is not None:
        quantile_metrics = quantile_loss(y_true, y_pred_low, y_pred_high)
        metrics.update(quantile_metrics)
    
    return metrics

def evaluate_stratified(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray, 
                       strata_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance stratified by categorical variable.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        strata: Array of stratum identifiers
        strata_names: List of stratum names (optional)
        
    Returns:
        Dictionary mapping stratum names to metric dictionaries
    """
    strata_metrics = {}
    unique_strata = np.unique(strata)
    
    for i, stratum in enumerate(unique_strata):
        mask = strata == stratum
        if np.sum(mask) > 0:  # Ensure there are samples in this stratum
            stratum_name = strata_names[i] if strata_names and i < len(strata_names) else str(stratum)
            stratum_metrics = calculate_metrics(y_true[mask], y_pred[mask])
            strata_metrics[stratum_name] = stratum_metrics
    
    return strata_metrics

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, y_pred_low: np.ndarray = None, 
                            y_pred_high: np.ndarray = None, save_path: str = None) -> plt.Figure:
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        y_pred_low: Array of lower bound predictions (optional)
        y_pred_high: Array of upper bound predictions (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
    
    # Add prediction intervals if provided
    if y_pred_low is not None and y_pred_high is not None:
        for i in range(len(y_true)):
            ax.plot([y_true[i], y_true[i]], [y_pred_low[i], y_pred_high[i]], 
                   color='red', alpha=0.2, linewidth=1)
        
        # Add a sample interval to the legend
        ax.plot([], [], color='red', alpha=0.5, linewidth=2, label='90% Prediction Interval')
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
           'k--', label='Perfect Prediction')
    
    # Set labels and title
    ax.set_xlabel('Actual Revenue (USD)')
    ax.set_ylabel('Predicted Revenue (USD)')
    ax.set_title('Actual vs Predicted Box Office Revenue')
    
    # Add a legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format axes with commas for thousands
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    # Set axis limits
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Ensure square aspect ratio
    ax.set_aspect('equal')
    
    # Add metrics text
    metrics = calculate_metrics(y_true, y_pred, y_pred_low, y_pred_high)
    metrics_text = (
        f"MAE: ${metrics['mae']:,.0f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"RMSE: ${metrics['rmse']:,.0f}"
    )
    if 'interval_coverage_percent' in metrics:
        metrics_text += f"\nInterval Coverage: {metrics['interval_coverage_percent']:.1f}%"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved actual vs predicted plot to {save_path}")
    
    return fig

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Create a histogram of prediction errors.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    errors = y_pred - y_true
    rel_errors = 100 * (errors / y_true)  # Percentage errors
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot absolute errors
    sns.histplot(errors, kde=True, ax=ax1)
    ax1.set_xlabel('Absolute Error (Predicted - Actual) in USD')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Errors')
    
    # Format x-axis with commas for thousands
    ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    # Add vertical line at zero
    ax1.axvline(x=0, color='red', linestyle='--')
    
    # Add metrics text
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics_text = (
        f"Mean Error: ${mean_error:,.0f}\n"
        f"Median Error: ${median_error:,.0f}\n"
        f"MAE: ${mae:,.0f}"
    )
    
    ax1.text(0.95, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot percentage errors
    sns.histplot(rel_errors, kde=True, ax=ax2)
    ax2.set_xlabel('Percentage Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Percentage Errors')
    
    # Add vertical line at zero
    ax2.axvline(x=0, color='red', linestyle='--')
    
    # Add metrics text
    mean_pct_error = np.mean(rel_errors)
    median_pct_error = np.median(rel_errors)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    metrics_text = (
        f"Mean % Error: {mean_pct_error:.2f}%\n"
        f"Median % Error: {median_pct_error:.2f}%\n"
        f"MAPE: {mape:.2f}%"
    )
    
    ax2.text(0.95, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")
    
    return fig

def plot_residuals_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Create a scatter plot of residuals vs predicted values.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    ax.scatter(y_pred, residuals, alpha=0.6, color='blue')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Predicted Revenue (USD)')
    ax.set_ylabel('Residuals (Actual - Predicted) in USD')
    ax.set_title('Residuals vs Predicted Values')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis with commas for thousands
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    # Add LOESS or moving average trend line to check for patterns
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # Sort for lowess
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        residuals_sorted = residuals[sorted_indices]
        
        # Fit lowess
        lowess_result = lowess(residuals_sorted, y_pred_sorted, frac=0.3)
        
        # Plot smoothed line
        ax.plot(lowess_result[:, 0], lowess_result[:, 1], color='red', 
               linewidth=2, label='LOWESS Trend')
        ax.legend()
    except ImportError:
        logger.warning("statsmodels not available for LOWESS smoothing")
        # Calculate moving average as alternative
        window_size = max(int(len(y_pred) * 0.1), 5)  # 10% of data points, minimum 5
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        residuals_sorted = residuals[sorted_indices]
        
        # Simple moving average
        moving_avg = np.convolve(residuals_sorted, np.ones(window_size)/window_size, mode='valid')
        x_vals = y_pred_sorted[window_size-1:][:len(moving_avg)]
        
        # Plot moving average
        ax.plot(x_vals, moving_avg, color='red', linewidth=2, label=f'Moving Avg (n={window_size})')
        ax.legend()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residuals vs predicted plot to {save_path}")
    
    return fig

def save_metrics_to_json(metrics: Dict[str, Any], save_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Convert numpy values to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    metrics_cleaned = convert_numpy(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_cleaned, f, indent=2)
    
    logger.info(f"Saved metrics to {save_path}")
