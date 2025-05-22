# app/evaluation/model_tracker.py
"""
Model tracking and evaluation module for the BORP project.
"""
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import shutil
from pathlib import Path

from app.utils.logging import get_logger
from app.evaluation.metrics import (
    calculate_metrics, 
    evaluate_stratified, 
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_residuals_vs_predicted,
    save_metrics_to_json
)

logger = get_logger("filmquant_ml.evaluation.model_tracker")

class ModelTracker:
    """
    Class for tracking model versions, parameters, and performance.
    
    This class is responsible for:
    - Saving model artifacts
    - Tracking hyperparameters
    - Recording evaluation metrics
    - Generating evaluation reports and visualizations
    - Comparing different model versions
    """
    
    def __init__(self, base_dir: str = None, experiment_name: str = None):
        """
        Initialize the model tracker.
        
        Args:
            base_dir: Base directory for storing model tracking information
            experiment_name: Name of the experiment
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), "model_tracking")
        self.experiment_name = experiment_name or f"filmquant_ml_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory structure
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        
        self._create_directory_structure()
        logger.info(f"Initialized model tracker for experiment: {self.experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _create_directory_structure(self) -> None:
        """Create the necessary directory structure for the experiment."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.debug(f"Created directory structure for experiment: {self.experiment_name}")
    
    def log_hyperparameters(self, model_version: str, hyperparameters: Dict[str, Any]) -> str:
        """
        Log hyperparameters for a model version.
        
        Args:
            model_version: Version identifier for the model
            hyperparameters: Dictionary of hyperparameters
            
        Returns:
            Path to the saved hyperparameters file
        """
        model_dir = os.path.join(self.models_dir, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Add timestamp
        hyperparameters['timestamp'] = datetime.now().isoformat()
        hyperparameters['model_version'] = model_version
        
        # Save hyperparameters
        hp_file = os.path.join(model_dir, "hyperparameters.json")
        with open(hp_file, 'w') as f:
            json.dump(hyperparameters, f, indent=2)
        
        logger.info(f"Logged hyperparameters for model version {model_version}")
        return hp_file
    
    def log_model_files(self, model_version: str, model_files: Dict[str, str]) -> Dict[str, str]:
        """
        Copy model files to the tracking directory.
        
        Args:
            model_version: Version identifier for the model
            model_files: Dictionary mapping file types to file paths
            
        Returns:
            Dictionary mapping file types to tracked file paths
        """
        model_dir = os.path.join(self.models_dir, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        tracked_files = {}
        
        for file_type, file_path in model_files.items():
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found: {file_path}")
                continue
            
            # Determine destination file path
            dest_filename = os.path.basename(file_path)
            dest_path = os.path.join(model_dir, dest_filename)
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            tracked_files[file_type] = dest_path
            
            logger.info(f"Copied {file_type} file from {file_path} to {dest_path}")
        
        # Save file mapping
        file_mapping = os.path.join(model_dir, "file_mapping.json")
        with open(file_mapping, 'w') as f:
            json.dump(tracked_files, f, indent=2)
        
        logger.info(f"Logged model files for model version {model_version}")
        return tracked_files
    
    def evaluate_model(self, model_version: str, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_low: np.ndarray = None, y_pred_high: np.ndarray = None,
                      feature_names: List[str] = None, feature_importance: Dict[str, float] = None,
                      strata: Dict[str, np.ndarray] = None, metadata: Dict[str, Any] = None,
                      generate_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate a model and record metrics.
        
        Args:
            model_version: Version identifier for the model
            y_true: Array of true target values
            y_pred: Array of predicted values
            y_pred_low: Array of lower bound predictions (optional)
            y_pred_high: Array of upper bound predictions (optional)
            feature_names: List of feature names (optional)
            feature_importance: Dictionary mapping feature names to importance scores (optional)
            strata: Dictionary mapping stratum names to boolean masks for stratified evaluation (optional)
            metadata: Additional metadata to include in the evaluation report (optional)
            generate_plots: Whether to generate evaluation plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model version {model_version}")
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_pred_low, y_pred_high)
        
        # Add metadata
        if metadata:
            metrics['metadata'] = metadata
        
        # Add feature importance if provided
        if feature_names and feature_importance:
            # Ensure feature_importance is sorted by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            metrics['feature_importance'] = [{"feature": k, "importance": v} for k, v in sorted_importance]
        
        # Calculate stratified metrics if provided
        if strata:
            stratified_metrics = {}
            for stratum_name, stratum_mask in strata.items():
                if np.sum(stratum_mask) > 0:  # Ensure there are samples in this stratum
                    stratum_metrics = calculate_metrics(
                        y_true[stratum_mask], 
                        y_pred[stratum_mask],
                        y_pred_low[stratum_mask] if y_pred_low is not None else None,
                        y_pred_high[stratum_mask] if y_pred_high is not None else None
                    )
                    stratified_metrics[stratum_name] = stratum_metrics
            
            metrics['stratified_metrics'] = stratified_metrics
        
        # Save metrics
        metrics_file = os.path.join(self.metrics_dir, f"{model_version}_metrics.json")
        save_metrics_to_json(metrics, metrics_file)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Generate plots if requested
        if generate_plots:
            plots_dir = os.path.join(self.plots_dir, model_version)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Actual vs Predicted plot
            actual_vs_pred_path = os.path.join(plots_dir, "actual_vs_predicted.png")
            plot_actual_vs_predicted(y_true, y_pred, y_pred_low, y_pred_high, actual_vs_pred_path)
            
            # Error distribution plot
            error_dist_path = os.path.join(plots_dir, "error_distribution.png")
            plot_error_distribution(y_true, y_pred, error_dist_path)
            
            # Residuals vs Predicted plot
            residuals_path = os.path.join(plots_dir, "residuals_vs_predicted.png")
            plot_residuals_vs_predicted(y_true, y_pred, residuals_path)
            
            # Feature importance plot if provided
            if feature_importance and len(feature_importance) > 0:
                importance_path = os.path.join(plots_dir, "feature_importance.png")
                self._plot_feature_importance(feature_importance, importance_path)
            
            logger.info(f"Generated evaluation plots in {plots_dir}")
        
        return metrics
    
    def _plot_feature_importance(self, feature_importance: Dict[str, float], save_path: str) -> plt.Figure:
        """
        Create a bar plot of feature importance scores.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_importance)
        
        # Limit to top 20 features
        if len(features) > 20:
            features = features[:20]
            importance = importance[:20]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importance, align='center')
        
        # Set labels
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.05, i, f"{width:.3f}", 
                   ha='left', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
        
        return fig
    
    def compare_models(self, model_versions: List[str]) -> pd.DataFrame:
        """
        Compare metrics across multiple model versions.
        
        Args:
            model_versions: List of model version identifiers to compare
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_version in model_versions:
            metrics_file = os.path.join(self.metrics_dir, f"{model_version}_metrics.json")
            
            if not os.path.exists(metrics_file):
                logger.warning(f"Metrics file not found for model version {model_version}")
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Add model version to metrics
            metrics['model_version'] = model_version
            
            # Extract hyperparameters if available
            hp_file = os.path.join(self.models_dir, model_version, "hyperparameters.json")
            if os.path.exists(hp_file):
                with open(hp_file, 'r') as f:
                    hyperparameters = json.load(f)
                
                # Add key hyperparameters
                for k, v in hyperparameters.items():
                    if k not in ['timestamp', 'model_version']:
                        metrics[f"hp_{k}"] = v
            
            comparison_data.append(metrics)
        
        # Convert to DataFrame
        if not comparison_data:
            logger.warning("No valid metrics found for comparison")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Flatten nested dictionaries
        if 'stratified_metrics' in df.columns:
            df = df.drop(columns=['stratified_metrics'])
        
        if 'metadata' in df.columns:
            df = df.drop(columns=['metadata'])
        
        # Drop complex columns
        for col in df.columns:
            if isinstance(df[col].iloc[0], (list, dict)):
                df = df.drop(columns=[col])
        
        # Sort by MAE (assuming lower is better)
        if 'mae' in df.columns:
            df = df.sort_values('mae')
        
        logger.info(f"Generated comparison of {len(df)} model versions")
        return df
    
    def generate_report(self, model_version: str, output_format: str = 'json') -> Dict[str, Any]:
        """
        Generate a comprehensive report for a model version.
        
        Args:
            model_version: Version identifier for the model
            output_format: Format for the report ('json' or 'html')
            
        Returns:
            Dictionary with report information
        """
        # Initialize report
        report = {"model_version": model_version}
        
        # Load metrics
        metrics_file = os.path.join(self.metrics_dir, f"{model_version}_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            report.update(metrics)
        else:
            logger.warning(f"Metrics file not found for model version {model_version}")
        
        # Load hyperparameters
        hp_file = os.path.join(self.models_dir, model_version, "hyperparameters.json")
        if os.path.exists(hp_file):
            with open(hp_file, 'r') as f:
                hyperparameters = json.load(f)
            report["hyperparameters"] = hyperparameters
        else:
            logger.warning(f"Hyperparameters file not found for model version {model_version}")
        
        # Get list of model files
        model_dir = os.path.join(self.models_dir, model_version)
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            report["model_files"] = model_files
        else:
            logger.warning(f"Model directory not found for model version {model_version}")
        
        # Get list of plots
        plots_dir = os.path.join(self.plots_dir, model_version)
        if os.path.exists(plots_dir):
            plots = os.listdir(plots_dir)
            report["plots"] = plots
        else:
            logger.warning(f"Plots directory not found for model version {model_version}")
        
        # Save report
        report_file = os.path.join(self.experiment_dir, f"{model_version}_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated report for model version {model_version}")
        
        # Generate HTML report if requested
        if output_format == 'html':
            html_report = self._generate_html_report(report, model_version)
            html_file = os.path.join(self.experiment_dir, f"{model_version}_report.html")
            with open(html_file, 'w') as f:
                f.write(html_report)
            logger.info(f"Generated HTML report for model version {model_version}")
        
        return report
    
    def _generate_html_report(self, report: Dict[str, Any], model_version: str) -> str:
        """
        Generate an HTML report from a report dictionary.
        
        Args:
            report: Dictionary with report information
            model_version: Version identifier for the model
            
        Returns:
            HTML string
        """
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>BORP Model Report - {model_version}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #3498db; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".metric-card { background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }",
            ".metric-name { font-size: 14px; color: #7f8c8d; }",
            ".plot-container { margin: 20px 0; text-align: center; }",
            ".plot-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>BORP Model Report - Version: {model_version}</h1>"
        ]
        
        # Add timestamp
        if "timestamp" in report:
            html.append(f"<p>Generated on: {report['timestamp']}</p>")
        
        # Add key metrics section
        html.append("<h2>Key Metrics</h2>")
        html.append("<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>")
        
        key_metrics = ["mae", "mape", "rmse"]
        if all(metric in report for metric in key_metrics):
            for metric in key_metrics:
                html.append(f"<div class='metric-card' style='flex: 1; min-width: 200px;'>")
                html.append(f"<div class='metric-name'>{metric.upper()}</div>")
                
                # Format metric value
                value = report[metric]
                formatted_value = f"${value:,.2f}" if metric in ["mae", "rmse"] else f"{value:.2f}%"
                
                html.append(f"<div class='metric-value'>{formatted_value}</div>")
                html.append("</div>")
        
        html.append("</div>")
        
        # Add interval coverage if available
        if "interval_coverage_percent" in report:
            html.append("<h3>Prediction Interval</h3>")
            html.append("<div class='metric-card'>")
            html.append("<div class='metric-name'>90% Prediction Interval Coverage</div>")
            html.append(f"<div class='metric-value'>{report['interval_coverage_percent']:.2f}%</div>")
            html.append("</div>")
        
        # Add plots section
        plots_dir = os.path.join(self.plots_dir, model_version)
        if os.path.exists(plots_dir) and os.listdir(plots_dir):
            html.append("<h2>Evaluation Plots</h2>")
            
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith(".png"):
                    plot_path = f"../plots/{model_version}/{plot_file}"
                    plot_title = plot_file.replace(".png", "").replace("_", " ").title()
                    
                    html.append("<div class='plot-container'>")
                    html.append(f"<h3>{plot_title}</h3>")
                    html.append(f"<img src='{plot_path}' alt='{plot_title}'>")
                    html.append("</div>")
        
        # Add hyperparameters section
        if "hyperparameters" in report:
            html.append("<h2>Hyperparameters</h2>")
            html.append("<table>")
            html.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            for param, value in report["hyperparameters"].items():
                if param not in ["timestamp", "model_version"]:
                    html.append(f"<tr><td>{param}</td><td>{value}</td></tr>")
            
            html.append("</table>")
        
        # Add feature importance section
        if "feature_importance" in report:
            html.append("<h2>Feature Importance</h2>")
            html.append("<table>")
            html.append("<tr><th>Feature</th><th>Importance</th></tr>")
            
            for feature in report["feature_importance"]:
                html.append(f"<tr><td>{feature['feature']}</td><td>{feature['importance']:.4f}</td></tr>")
            
            html.append("</table>")
        
        # Add stratified metrics section
        if "stratified_metrics" in report:
            html.append("<h2>Stratified Metrics</h2>")
            
            for stratum, metrics in report["stratified_metrics"].items():
                html.append(f"<h3>{stratum}</h3>")
                html.append("<table>")
                html.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for metric, value in metrics.items():
                    formatted_value = f"${value:,.2f}" if metric in ["mae", "rmse"] else f"{value:.2f}"
                    if metric.endswith("_percent"):
                        formatted_value = f"{value:.2f}%"
                    
                    html.append(f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>")
                
                html.append("</table>")
        
        # Add model files section
        if "model_files" in report:
            html.append("<h2>Model Files</h2>")
            html.append("<ul>")
            
            for file in report["model_files"]:
                html.append(f"<li>{file}</li>")
            
            html.append("</ul>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def list_model_versions(self) -> List[str]:
        """
        List all tracked model versions.
        
        Returns:
            List of model version identifiers
        """
        if not os.path.exists(self.models_dir):
            return []
        
        return [d for d in os.listdir(self.models_dir) 
                if os.path.isdir(os.path.join(self.models_dir, d))]
    
    def get_best_model_version(self, metric: str = "mae", higher_is_better: bool = False) -> Optional[str]:
        """
        Get the best model version based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher values are better
            
        Returns:
            Best model version or None if no models found
        """
        model_versions = self.list_model_versions()
        
        if not model_versions:
            logger.warning("No model versions found")
            return None
        
        models_metrics = []
        
        for model_version in model_versions:
            metrics_file = os.path.join(self.metrics_dir, f"{model_version}_metrics.json")
            
            if not os.path.exists(metrics_file):
                logger.warning(f"Metrics file not found for model version {model_version}")
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if metric not in metrics:
                logger.warning(f"Metric '{metric}' not found for model version {model_version}")
                continue
            
            models_metrics.append((model_version, metrics[metric]))
        
        if not models_metrics:
            logger.warning(f"No models found with metric '{metric}'")
            return None
        
        # Sort by metric (ascending or descending based on higher_is_better)
        models_metrics.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        best_model = models_metrics[0][0]
        logger.info(f"Best model version by {metric}: {best_model}")
        
        return best_model
