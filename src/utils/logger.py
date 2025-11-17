import logging
import os
from src.core.config import Config

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with both console and file handlers.
    Each module can call get_logger(__name__) to get its own logger.
    """
    cfg = Config()
    logger = logging.getLogger(name)

    if not logger.handlers:  # Prevent duplicate handlers
        # Ensure logs directory exists
        log_dir = cfg.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        log_file_path = os.path.join(log_dir, "pipeline.log")
        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
