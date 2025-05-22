# app/evaluation/visualization.py
"""
Visualization module for model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Union, Tuple

from app.utils.logging import get_logger

logger = get_logger("filmquant_ml.evaluation.visualization")

def set_plot_style():
    """Set consistent style for all plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def plot_metrics_comparison(metrics_dict: Dict[str, List[float]], 
                           model_names: List[str], 
                           plot_title: str = "Model Comparison",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart comparing metrics across multiple models.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and lists of values for each model
        model_names: Names of the models being compared
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Determine the number of metrics and models
    n_metrics = len(metrics_dict)
    n_models = len(model_names)
    
    # Set up the figure
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    # Handle the case of a single metric
    if n_metrics == 1:
        axes = [axes]
    
    # Set bar width
    bar_width = 0.8 / n_models
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[i]
        
        # Plot bars for each model
        for j, (model_name, value) in enumerate(zip(model_names, values)):
            x_pos = j * bar_width
            ax.bar(x_pos, value, width=bar_width, label=model_name if i == 0 else "")
        
        # Set axis labels and title
        ax.set_title(metric_name.upper())
        ax.set_ylabel("Value")
        
        # Set x-axis ticks
        x_ticks = np.arange(0, n_models * bar_width, bar_width)
        ax.set_xticks(x_ticks + bar_width / 2)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        
        # Add value labels on top of each bar
        for j, value in enumerate(values):
            ax.text(j * bar_width + bar_width / 2, value * 1.02, f"{value:.2f}", 
                   ha='center', va='bottom', fontsize=10)
    
    # Add legend to the first axis only if there are multiple models
    if n_models > 1:
        axes[0].legend(bbox_to_anchor=(0, -0.2), loc="upper left")
    
    # Set main title
    fig.suptitle(plot_title, fontsize=18)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison plot to {save_path}")
    
    return fig

def plot_metric_history(metric_values: List[float], model_versions: List[str],
                       metric_name: str = "MAE", 
                       plot_title: str = "Metric History",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart showing the trend of a metric across model versions.
    
    Args:
        metric_values: List of metric values for each model version
        model_versions: List of model version identifiers
        metric_name: Name of the metric being plotted
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line
    ax.plot(model_versions, metric_values, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Add labels for each point
    for i, (version, value) in enumerate(zip(model_versions, metric_values)):
        ax.text(i, value * 1.02, f"{value:.2f}", ha='center')
    
    # Add trendline
    if len(model_versions) > 1:
        z = np.polyfit(range(len(model_versions)), metric_values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(model_versions)), p(range(len(model_versions))), 
               linestyle='--', color='red', label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
        ax.legend()
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel("Model Version")
    ax.set_ylabel(metric_name)
    
    # Set x-axis ticks
    ax.set_xticks(range(len(model_versions)))
    ax.set_xticklabels(model_versions, rotation=45, ha="right")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metric history plot to {save_path}")
    
    return fig

def plot_feature_importance_comparison(feature_importances: Dict[str, Dict[str, float]], 
                                     top_n: int = 10,
                                     plot_title: str = "Feature Importance Comparison",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart comparing feature importances across multiple models.
    
    Args:
        feature_importances: Dictionary mapping model names to feature importance dictionaries
        top_n: Number of top features to show
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Collect all unique features across models
    all_features = set()
    for model_importances in feature_importances.values():
        all_features.update(model_importances.keys())
    
    # Create a DataFrame with all features and importances
    df = pd.DataFrame(index=list(all_features))
    
    for model_name, model_importances in feature_importances.items():
        df[model_name] = pd.Series(model_importances)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Calculate average importance for sorting
    df['avg_importance'] = df.mean(axis=1)
    
    # Sort by average importance and select top N features
    df = df.sort_values('avg_importance', ascending=False).head(top_n)
    df = df.drop(columns=['avg_importance'])
    
    # Transpose for plotting
    df_plot = df.T
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(df_plot, annot=True, cmap='viridis', fmt=".3f", linewidths=.5, ax=ax)
    
    # Set title and labels
    ax.set_title(plot_title)
    ax.set_ylabel("Model")
    ax.set_xlabel("Feature")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance comparison plot to {save_path}")
    
    return fig

def plot_prediction_interval_coverage(coverage_values: List[float], model_versions: List[str],
                                    target_coverage: float = 90.0,
                                    plot_title: str = "Prediction Interval Coverage",
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart showing prediction interval coverage across model versions.
    
    Args:
        coverage_values: List of coverage percentages for each model version
        model_versions: List of model version identifiers
        target_coverage: Target coverage percentage (usually 90%)
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line
    ax.plot(model_versions, coverage_values, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Add target coverage line
    ax.axhline(y=target_coverage, color='red', linestyle='--', label=f"Target ({target_coverage}%)")
    
    # Add labels for each point
    for i, (version, value) in enumerate(zip(model_versions, coverage_values)):
        ax.text(i, value + 1, f"{value:.1f}%", ha='center')
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel("Model Version")
    ax.set_ylabel("Coverage (%)")
    
    # Set y-axis limits
    min_coverage = min(min(coverage_values), target_coverage) - 5
    max_coverage = max(max(coverage_values), target_coverage) + 5
    ax.set_ylim(max(0, min_coverage), min(100, max_coverage))
    
    # Set x-axis ticks
    ax.set_xticks(range(len(model_versions)))
    ax.set_xticklabels(model_versions, rotation=45, ha="right")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction interval coverage plot to {save_path}")
    
    return fig

def plot_stratified_metrics(stratified_metrics: Dict[str, Dict[str, float]],
                          metric_name: str = "mae",
                          plot_title: str = "Stratified Metrics",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart showing a metric value across different strata.
    
    Args:
        stratified_metrics: Dictionary mapping stratum names to metric dictionaries
        metric_name: Name of the metric to plot
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Extract metric values for each stratum
    strata = []
    values = []
    
    for stratum, metrics in stratified_metrics.items():
        if metric_name in metrics:
            strata.append(stratum)
            values.append(metrics[metric_name])
    
    # Sort by metric value for better visualization
    sorted_indices = np.argsort(values)
    strata = [strata[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot horizontal bars
    y_pos = np.arange(len(strata))
    ax.barh(y_pos, values, height=0.6)
    
    # Add labels for each bar
    for i, value in enumerate(values):
        ax.text(value + value * 0.02, i, f"{value:.2f}", va='center')
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel(f"{metric_name.upper()} Value")
    ax.set_ylabel("Stratum")
    
    # Set y-axis ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strata)
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stratified metrics plot to {save_path}")
    
    return fig

def plot_hyperparameter_importance(hp_importances: Dict[str, float],
                                 plot_title: str = "Hyperparameter Importance",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart showing hyperparameter importance.
    
    Args:
        hp_importances: Dictionary mapping hyperparameter names to importance scores
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Sort by importance
    sorted_items = sorted(hp_importances.items(), key=lambda x: x[1], reverse=True)
    hp_names, importance_values = zip(*sorted_items)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    y_pos = np.arange(len(hp_names))
    ax.barh(y_pos, importance_values, height=0.6)
    
    # Add labels for each bar
    for i, value in enumerate(importance_values):
        ax.text(value + value * 0.02, i, f"{value:.3f}", va='center')
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Hyperparameter")
    
    # Set y-axis ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hp_names)
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hyperparameter importance plot to {save_path}")
    
    return fig

def plot_learning_curve(train_scores: List[float], val_scores: List[float], train_sizes: List[int],
                       metric_name: str = "MAE",
                       plot_title: str = "Learning Curve",
                       higher_is_better: bool = False,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart showing the learning curve (train/validation scores vs training set size).
    
    Args:
        train_scores: List of training scores for different training set sizes
        val_scores: List of validation scores for different training set sizes
        train_sizes: List of training set sizes
        metric_name: Name of the metric being plotted
        plot_title: Title for the plot
        higher_is_better: Whether higher metric values are better
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(train_sizes, train_scores, marker='o', linestyle='-', color='blue', label='Training Score')
    ax.plot(train_sizes, val_scores, marker='s', linestyle='-', color='red', label='Validation Score')
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric_name)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust y-axis limits if higher is better
    if higher_is_better:
        # Ensure y-axis starts at 0 or the minimum score, whichever is lower
        ax.set_ylim(bottom=min(0, min(min(train_scores), min(val_scores)) * 0.95))
    else:
        # Ensure y-axis starts at 0 for metrics where lower is better
        ax.set_ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curve plot to {save_path}")
    
    return fig

def plot_calibration_curve(pred_probabilities: List[float], observed_frequencies: List[float],
                          plot_title: str = "Calibration Curve",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart showing model calibration (predicted probabilities vs observed frequencies).
    
    Args:
        pred_probabilities: List of predicted probabilities
        observed_frequencies: List of observed frequencies
        plot_title: Title for the plot
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot(pred_probabilities, observed_frequencies, marker='o', linestyle='-', 
           color='blue', label='Calibration Curve')
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
    
    # Set axis labels and title
    ax.set_title(plot_title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Make plot square
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve plot to {save_path}")
    
    return fig
