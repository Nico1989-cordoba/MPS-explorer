"""
Logging Configuration Module for MPS Explorer

Provides centralized logging setup with:
- Console and file output
- Rotating file handlers (prevents huge log files)
- Configurable log levels
- Structured formatting with timestamps and context
- Stack traces for debugging exceptions

Usage:
    from logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging system with console and optional file output.

    Parameters
    ----------
    log_level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Optional[str], default None
        Path to log file. If None, logs to console only.
        If provided, logs to both console and file.
    max_bytes : int, default 10_485_760 (10 MB)
        Maximum size of log file before rotation.
    backup_count : int, default 5
        Number of backup log files to keep.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Rotating file handler ensures log files don't grow unbounded.
    When max_bytes is reached, file is rotated and renamed.
    Example: app.log → app.log.1, app.log.2, etc.

    Examples
    --------
    >>> logger = setup_logging(log_level="DEBUG", log_file="app.log")
    >>> logger.debug("Detailed information")
    >>> logger.info("Important information")
    >>> logger.warning("Warning message")
    >>> logger.error("Error occurred")
    """
    # Get or create logger
    logger = logging.getLogger("MPS_explorer")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Define formatter with timestamp, level, logger name, and message
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with Rotation (if log_file provided)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler: rotates when file exceeds max_bytes
        # Keeps backup_count old files (e.g., app.log.1, app.log.2, etc.)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(
            f"Logging to file: {log_file} (max {max_bytes / 1_000_000:.1f} MB, {backup_count} backups)"
        )

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance by name.

    Parameters
    ----------
    name : Optional[str], default None
        Logger name (typically __name__ of calling module).
        If None, returns root logger.

    Returns
    -------
    logging.Logger
        Logger instance configured by setup_logging().

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Message from this module")
    """
    if name is None:
        return logging.getLogger("MPS_explorer")
    return logging.getLogger(name)


def load_logging_config(config_file: str) -> Dict[str, Any]:
    """
    Load logging configuration from JSON file.

    Parameters
    ----------
    config_file : str
        Path to JSON config file with logging settings.

    Returns
    -------
    dict
        Configuration dictionary with keys:
        - log_level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        - log_file: path to log file or null
        - max_bytes: max file size in bytes
        - backup_count: number of backup files

    Examples
    --------
    >>> config = load_logging_config("logging_config.json")
    >>> logger = setup_logging(**config)
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            # json.load returns Any; annotate explicitly so mypy accepts it
            # against the declared Dict[str, Any] return type.
            config: Dict[str, Any] = json.load(f)
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {config_file}: {e}")
        return {
            "log_level": "INFO",
            "log_file": None,
            "max_bytes": 10_485_760,
            "backup_count": 5,
        }


def get_log_filename(base_name: str = "mps_explorer") -> str:
    """
    Generate log filename with timestamp.

    Parameters
    ----------
    base_name : str, default "mps_explorer"
        Base name for log file.

    Returns
    -------
    str
        Log filename with timestamp, e.g., "mps_explorer_2026-05-28.log"

    Examples
    --------
    >>> log_file = get_log_filename()
    >>> print(log_file)
    mps_explorer_2026-05-28.log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    return f"{base_name}_{timestamp}.log"


# Module-level convenience function
# Usage: from logging_config import logger
logger = logging.getLogger("MPS_explorer")
