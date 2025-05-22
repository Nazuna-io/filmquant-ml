# app/utils/logging.py
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
import sys
from typing import Optional, Dict, Any
from app.config import settings

def configure_logging(
    app_name: Optional[str] = None,
    log_level: Optional[int] = None,
    log_format: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    log_config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up application logging with flexible configuration options.
    
    Args:
        app_name: Name of the application/logger
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Custom log format string, if None uses a default format
        log_dir: Directory for log files
        log_to_console: Whether to output logs to console
        log_to_file: Whether to write logs to file
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        log_config: Additional configuration options
        
    Returns:
        The configured logger instance
    """
    # Use provided values or get from config
    app_name = app_name or settings.app.name.lower().replace(" ", "_")
    log_level_str = log_level or settings.logging.level
    log_format = log_format or settings.logging.format
    log_dir = log_dir or settings.logging.directory
    log_to_console = log_to_console if log_to_console is not None else settings.logging.console
    log_to_file = log_to_file if log_to_file is not None else settings.logging.file
    max_bytes = max_bytes or settings.logging.max_bytes
    backup_count = backup_count or settings.logging.backup_count
    
    # Convert string log level to int
    if isinstance(log_level_str, str):
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    else:
        log_level = log_level_str
    
    # Create log directory if needed
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Use custom format or default
    if log_format is None:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s'
        )
    else:
        formatter = logging.Formatter(log_format)
    
    # Add file handler
    if log_to_file:
        # Determine the type of file handler based on config
        if log_config and log_config.get("time_based_rotation", False):
            # Time-based rotation (e.g., daily)
            file_handler = TimedRotatingFileHandler(
                filename=f"{log_dir}/{app_name}.log",
                when=log_config.get("rotation_when", "midnight"),
                interval=log_config.get("rotation_interval", 1),
                backupCount=backup_count
            )
        else:
            # Size-based rotation
            file_handler = RotatingFileHandler(
                filename=f"{log_dir}/{app_name}.log",
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log startup message
    logger.info(f"Logging initialized for '{app_name}' at level {logging.getLevelName(log_level)}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    If the logger doesn't exist, it's created with the base configuration.
    
    Args:
        name: Name of the logger (typically module name)
        
    Returns:
        A logger instance
    """
    # Return the named logger
    return logging.getLogger(name)
