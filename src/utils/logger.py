"""
Centralized logging setup using loguru.

This module provides a setup_logger function that configures loguru
with customizable settings. Import and call setup_logger() once at the
start of your application, then use loguru's logger throughout your code.

Example usage:
    # In your main file or initialization:
    from src.utils.logger import setup_logger
    setup_logger(level="INFO", log_to_file=True)

    # In any other file:
    from loguru import logger
    logger.info("This is a log message")
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_to_file: bool = True,
    log_dir: str = "logs",
    log_filename: str = "app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
    format_string: Optional[str] = None,
    diagnose: bool = True,
) -> None:
    """
    Configure loguru logger with custom settings.

    Args:
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        log_to_file: Whether to log to a file in addition to console
        log_dir: Directory to store log files
        log_filename: Name of the log file
        rotation: When to rotate the log file (e.g., "10 MB", "1 day", "00:00")
        retention: How long to keep old log files (e.g., "7 days", "10 files")
        compression: Compression format for rotated logs (e.g., "zip", "gz")
        format_string: Custom format string for log messages
        diagnose: Whether to include diagnosis information in logs

    Example:
        >>> # Console shows INFO+, file logs DEBUG+
        >>> setup_logger(console_level="INFO", file_level="DEBUG", log_to_file=True)
        >>> from loguru import logger
        >>> logger.debug("This appears in file only")
        >>> logger.info("This appears in both console and file")
    """
    # Remove default handler
    logger.remove()

    # Default format if not provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=format_string,
        level=console_level,
        colorize=True,
        diagnose=diagnose,
    )

    # Add file handler if requested
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path / log_filename,
            format=format_string,
            level=file_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            diagnose=diagnose,
        )

    # Log all configuration parameters
    logger.info(
        f"Logger initialized with config: "
        f"console_level={console_level}, "
        f"file_level={file_level}, "
        f"log_to_file={log_to_file}, "
        f"log_dir={log_dir}, "
        f"log_filename={log_filename}, "
        f"rotation={rotation}, "
        f"retention={retention}, "
        f"compression={compression}, "
        f"format_string={'<custom>' if format_string else '<default>'}, "
        f"diagnose={diagnose}"
    )


def get_logger():
    """
    Get the configured logger instance.

    This is a convenience function that returns the loguru logger.
    You can also directly import logger from loguru after setup.

    Returns:
        The loguru logger instance

    Example:
        >>> from src.utils.logger import setup_logger, get_logger
        >>> setup_logger()
        >>> log = get_logger()
        >>> log.info("Using get_logger()")
    """
    return logger
